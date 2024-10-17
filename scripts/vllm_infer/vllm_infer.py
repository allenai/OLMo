"""
General script/tooling to:
1. Take in a set of .jsonl files (s3 or weka or whatever)
2. Load a model (huggingface or on weak or whatever)
3. Run inference on 'text' field of each json within the dataset and add the generated text to the 'output' field of the dataset

and let's just assume that I can run everything 
And have everything be runnable on a single node/single GPU
(and distribute in another way, using beaker-gantry experiments for each)
"""

import os
import json
import gzip
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
import glob
import time



# ========================================================
# =                   VLLM-OLMO STUFF                    =
# ========================================================
from typing import Iterable, List, Optional, Tuple, Union


from hf_olmo import *
from transformers import AutoModelForCausalLM

import torch
from torch import nn
from transformers import OlmoConfig
from hf_olmo import OLMoConfig

from vllm import LLM, SamplingParams
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors


class FlippedSiluAndMul(SiluAndMul):
    """OLMo is trained with SwiGLU with flipped halves."""

    def forward(self, x: torch.Tensor):
        a, b = x.chunk(2, dim=-1)
        flipped = torch.cat((b, a), dim=-1)
        return super().forward(flipped)

class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: OlmoConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tensor_model_parallel_world_size == 0

        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_sequence_length
        self.rope_theta = config.rope_theta
        self.clip_qkv = config.clip_qkv

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=config.include_bias,
            quant_config=quant_config,
        )

        if config.attention_layer_norm:
            # TODO: finish adding qk norm and norm_after
            self.k_norm = RMSNorm(
                (config.d_model // config.n_heads) * config.effective_n_kv_heads,
                eps=config.layer_norm_eps,
                #elementwise_affine=config.attention_layer_norm_with_affine,
                #bias=False,
            )
            self.q_norm = RMSNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )

        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=config.include_bias,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        #q = self.q_norm(q) 
        #k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: OlmoConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        try:
            self.intermediate_size = config.intermediate_size
        except AttributeError:
            if config.mlp_hidden_size is not None:
                self.intermediate_size = config.mlp_hidden_size // 2
            else:
                self.intermediate_size = (config.d_model * config.mlp_ratio) // 2

        # Feed-forward input projection.
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )

        # Activation function.
        self.act_fn = FlippedSiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OlmoDecoderLayer(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self,
                 config: OlmoConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        # Attention block.
        self.self_attn = OlmoAttention(config, cache_config, quant_config)

        # MLP block.
        self.mlp = OlmoMLP(config, quant_config)

        # LayerNorm

        self.norm_after = config.norm_after
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        """
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            elementwise_affine=False,
                                            bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     elementwise_affine=False,
                                                     bias=False)
        """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block.
        residual = hidden_states
        if self.norm_after:
            hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                           attn_metadata)
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                           attn_metadata)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        if self.norm_after:
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OlmoModel(nn.Module):

    def __init__(self,
                 config: Union[OlmoConfig, OLMoConfig],
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(config.embedding_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            OlmoDecoderLayer(config, cache_config, quant_config)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            #elementwise_affine=config.layer_norm_with_affine,
            #bias=config.bias_for_layer_norm
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # Apply blocks one-by-one.
        for layer_idx, decoder_layer in enumerate(self.layers):
            # shape: (batch_size, seq_len, d_model)
            hidden_states = decoder_layer(
                positions,
                hidden_states,
                kv_caches[layer_idx],
                attn_metadata,
            )

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class OlmoNewForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """

    def __init__(self,
                 config: OlmoConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.model = OlmoModel(config, cache_config, quant_config)
        if config.weight_tying:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                #self.unpadded_vocab_size,
                config.embedding_size,
                config.hidden_size,
                org_num_embeddings=config.embedding_size,
                #org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        self.logits_processor = LogitsProcessor(config.embedding_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def _create_map(self):
        mapper = {}
        for layer_i in range(self.config.n_layers):
            mapper[f"model.transformer.blocks.{layer_i}.att_proj.weight"] = f"model.layers.{layer_i}.self_attn.qkv_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.attn_out.weight"] = f"model.layers.{layer_i}.self_attn.o_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_proj.weight"] = f"model.layers.{layer_i}.mlp.gate_up_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_out.weight"] = f"model.layers.{layer_i}.mlp.down_proj.weight"

            mapper[f"model.transformer.blocks.{layer_i}.attn_norm.weight"] = f"model.layers.{layer_i}.input_layernorm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_norm.weight"] = f"model.layers.{layer_i}.post_attention_layernorm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.k_norm.weight"] = f"model.layers.{layer_i}.self_attn.k_norm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.q_norm.weight"] = f"model.layers.{layer_i}.self_attn.q_norm.weight"

        mapper["model.transformer.ln_f.weight"] = "model.norm.weight"
        mapper["model.transformer.wte.weight"] = "model.embed_tokens.weight"
        mapper["model.transformer.ff_out.weight"] = "lm_head.weight"
        return mapper

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mapper = self._create_map()
        print("mapper", mapper)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.weight_tying and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[mapper.get(name, name)]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)



# ========================================================
# =                   DATASET STUFF                      =
# ========================================================

def load_jsonl(input_path: str) -> list: 
    parsed_path = urlparse(input_path)
    if parsed_path.scheme == 's3':
        bucket_name = parsed_path.netloc
        object_key = parsed_path.path.lstrip('/')
        s3 = boto3.client('s3')
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            data = response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"The file does not exist: s3://{bucket_name}/{object_key}")
            else:
                raise
    else: # Local path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file does not exist: {input_path}")
        with open(input_path, 'rb') as f:
            data = f.read()
        print(f"File loaded locally: {input_path}")    
    if input_path.endswith('.gz'):
        data = gzip.decompress(data)
    data = [json.loads(_) for _ in data.splitlines()]      
    return data


def iter_jsonl(jsonl_list, batch_size):
    batch = []
    for el in jsonl_list:
        batch.append(el)
        if len(batch) == batch_size:
            yield batch 
            batch = []

    if batch:
        yield batch



def save_jsonl(output_dicts, output_path):
    output_data = b'\n'.join([json.dumps(_).encode('utf-8') for _ in output_dicts])
    if output_path.endswith('.gz'):
        output_data = gzip.compress(output_data)

    if output_path.startswith('s3://'):
        parsed_path = urlparse(output_path)
        bucket_name = parsed_path.netloc
        object_key = parsed_path.path.lstrip('/')
        
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=output_data)
        print(f"File saved to S3: s3://{bucket_name}/{object_key}")
    else:
        # Local path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f"File saved locally: {output_path}")


def check_exists(path) -> bool:
    # Checks if path exists either on local or on s3
    if path.startswith('s3://'):
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        key = parsed_path.path.lstrip('/')
        s3_client = boto3.client('s3')
        try:
            s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise e
    else:
        return os.path.isfile(path)


def input_path_to_output_path(input_path: str, input_dir: str, output_dir: str) -> str:
    # Replaces input_dir in input_path with output_dir
    return input_path.replace(input_dir, output_dir)


def list_files(input_dir: str, part_num: int, num_parts: int, ext='.jsonl') -> list:
    if input_dir.startswith('s3://'):
        parsed_uri = urlparse(input_dir)
        bucket_name = parsed_uri.netloc
        prefix = parsed_uri.path.lstrip('/')

        s3 = boto3.client('s3')
        matching_files = []

        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    if filename.endswith(ext):
                        matching_files.append(f"s3://{bucket_name}/{key}")
    else:
        search_path = os.path.join(input_dir, '**', '*' + ext)
        matching_files = glob.glob(search_path, recursive=True)

    return [f for i, f in enumerate(matching_files) if i % num_parts == part_num]


# ========================================================
# =                 MODEL LOADING STUFF                  =
# ========================================================

def load_peteish_model(checkpoint=None):
    if checkpoint == None:
        checkpoint = '/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646-hf'
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )    
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model("OLMoForCausalLM", OlmoNewForCausalLM)
    return LLM(model=checkpoint, trust_remote_code=True, gpu_memory_utilization=0.90)


# ========================================================
# =                  INFERENCE CODE                      =
# ========================================================

def process_batch(model: LLM, batch: List[dict], sampling_params: SamplingParams, prepend_eot: bool) -> List[dict]:
    if prepend_eot:
        for batch_el in batch:
            batch_el['text'] = '<|endoftext|>' + batch_el['text']

    input_texts = [_['text'] for _ in batch]
    outputs = model.generate(input_texts, sampling_params)
    for batch_el, output in zip(batch, outputs):
        batch_el['output_text'] = output.outputs[0].text
    return batch


def infer_jsonl(args):
    input_files = list_files(args.input_dir, args.part, args.num_parts)

    # Load model
    prepend_eot = args.prepend_eot
    if args.model_name == "peteish":
        # Do custom peteish load
        model = load_peteish_model(args.checkpoint)
        prepend_eot = True
    else:
        assert args.model_name == "meta-llama/Llama-3.1-8B"
        model = LLM(model=args.model_name) #

    start_time = time.time()
    processed = 0
    batch = []

    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=args.max_length)
    for input_file in input_files:
        print("Processing file: %s..." % input_file)
        output_path = input_path_to_output_path(input_file, args.input_dir, args.output_dir)
        if check_exists(output_path):
            print("%s already exists -- skipping" % output_path)
            continue
        input_data = load_jsonl(input_file)
        output_data = []
        for batch in iter_jsonl(input_data, args.batch_size):
            processed += len(batch)
            output_data.extend(process_batch(model, batch, sampling_params, prepend_eot=prepend_eot))
        save_jsonl(output_data, output_path)
    print("Processed %s files, %s prompts, in %.02f seconds" % 
          (len(input_files), processed, time.time() - start_time))



# ========================================================
# =                      MAIN BLOCK                      =
# ========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="One-off Inference for cLanguage Models")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B", help="Name of the model to use")
    parser.add_argument("--checkpoint", type=str, default=None, help="If extra checkpoint info is needed (e.g. for peteish models)")
    parser.add_argument("--input-dir", type=str, required=True, help="Input of directory that contains .jsonl.gz files")
    parser.add_argument("--part", type=int, default=0, help="If we partition the input dir's files into many pieces, which part here")
    parser.add_argument("--num-parts", type=int, default=1, help="If we partition the input dir's files into one piece, how many parts ehre")
    parser.add_argument("--output-dir", type=str, required=True, help="Where the outputs should go")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum length of generated text")
    parser.add_argument("--prepend-eot", type=bool, default=False, help="Whether or not to prepend <|endoftext|> to all inputs")
   
    args = parser.parse_args()
    infer_jsonl(args)







