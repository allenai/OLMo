import os
from auto_gptq.modeling._base import BaseGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.quantization.gptq import GPTQ
from auto_gptq.modeling._utils import get_module_by_name_prefix, get_module_by_name_suffix, get_device, CPU, CUDA_0, move_to_device, find_layers, pack_model, simple_dispatch_model, make_quant, make_sure_no_tensor_in_meta_device

# from auto_gptq.nn_modules.qlinear import Gen
from auto_gptq.utils.import_utils import TRITON_AVAILABLE
from safetensors.torch import save_file as safe_save
from accelerate.hooks import remove_hook_from_module
import accelerate
import torch
from typing import Optional, List, Dict, Union
import logging
import warnings

from olmo import Olmo

logger = logging.getLogger(__name__)


class OLMoGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str = None
    layers_block_name: str = None
    outside_layer_modules: List[str] = None
    inside_layer_modules: List[List[str]] = None
    lm_head_name: str = "lm_head"

    # TODO:
    # fused_attn_module_type: Optional[FusedBaseAttentionModule] = None
    # fused_mlp_module_type: Optional[FusedBaseMLPModule] = None

    # def __init__(
    #     self,
    #     model: torch.nn.Module,
    #     quantized: bool,
    #     quantize_config: BaseQuantizeConfig,
    #     is_triton_backend: bool = False,
    #     injected_fused_attention: bool = False,
    #     injected_fused_mlp: bool = False,
    #     trainable: bool = False
    # ):
    #     super().__init__()
    #
    #     self.model = model
    #     self.model_type = self.model.config.model_type
    #     self._quantized = quantized
    #     self.quantize_config = quantize_config
    #     self.config = self.model.config
    #
    #     self.is_triton_backend = is_triton_backend
    #     self.injected_fused_attention = injected_fused_attention
    #     self.injected_fused_mlp = injected_fused_mlp
    #     self.trainable = trainable

    @property
    def quantized(self):
        return getattr(self, "_quantized", False)
    #
    # @property
    # def hf_device_map(self):
    #     return getattr(self.model, "hf_device_map", None)
    #
    # @staticmethod
    # def _resize_attention_mask(attention_mask: List[torch.LongTensor]):
    #     return attention_mask
    #
    # @staticmethod
    # def _resize_position_ids(position_ids: List[torch.LongTensor]):
    #     return position_ids

    @torch.inference_mode()
    def quantize(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = True
    ):
        if self.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")
        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logger.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        examples = self._prepare_examples_for_quantization(examples, batch_size)

        class LayerHijacker(torch.nn.Module):
            """hijack layer's forward pass to cache data"""

            def __init__(self, m, device):
                super().__init__()
                self.module = m
                self.data_device = device if cache_examples_on_gpu else CPU

            def forward(self, inp=None, **kwargs):
                if inp is None:  # some models use all key-value arguments in forward pass call
                    for kwarg_name in ["hidden_states"]:
                        if kwarg_name in kwargs:
                            inp = kwargs[kwarg_name]
                            break
                layer_inputs.append(move_to_device(inp, self.data_device))

                # TODO: attention_mask as input.
                attention_masks.append(kwargs["attention_mask"].to(self.data_device))
                pos_ids = kwargs.get("position_ids", None)
                if pos_ids is not None:
                    position_ids.append(move_to_device(pos_ids, self.data_device))
                one_kwargs = dict()
                for k, v in kwargs.items():  # make sure other arguments also be captured
                    if k not in ["hidden_states", "attention_mask", "position_ids"]:
                        if isinstance(v, torch.Tensor):
                            one_kwargs[k] = move_to_device(v, self.data_device)
                        else:
                            one_kwargs[k] = v
                layer_input_kwargs.append(one_kwargs)
                raise ValueError

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = get_module_by_name_prefix(self.model, self.layers_block_name)

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == CPU:
            layers[0] = layers[0].to(CUDA_0)
            force_layer_back_to_cpu = True

        cur_layer_device = get_device(layers[0])
        ori_outside_layer_module_devices = {}
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to_device(module, cur_layer_device)

        # get inputs for first layer
        layers[0] = LayerHijacker(layers[0], cur_layer_device)
        for example in examples:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to_device(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        layers[0] = layers[0].module

        move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to_device(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        # resize attention mask and position ids for some special models
        attention_masks = self._resize_attention_mask(attention_masks)
        position_ids = self._resize_position_ids(position_ids)

        inside_layer_modules = self.inside_layer_modules
        if not self.quantize_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        quantizers = {}
        for i in range(len(layers)):
            logger.info(f"Start quantizing layer {i + 1}/{len(layers)}")
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, CUDA_0)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = find_layers(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        self.quantize_config.bits,
                        perchannel=True,
                        sym=self.quantize_config.sym,
                        mse=False,
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                    layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                    additional_layer_inputs = {
                        "attention_mask": layer_attention_mask
                    }
                    layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                    if layer_position_ids is not None:
                        additional_layer_inputs["position_ids"] = layer_position_ids
                    for k, v in layer_input_kwargs[j].items():
                        if isinstance(v, torch.Tensor):
                            additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                        else:
                            additional_layer_inputs[k] = v
                    layer(layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f'Quantizing {name} in layer {i + 1}/{len(layers)}...')
                    scale, zero, g_idx = gptq[name].fasterquant(
                        percdamp=self.quantize_config.damp_percent,
                        group_size=self.quantize_config.group_size,
                        actorder=self.quantize_config.desc_act
                    )
                    quantizers[f'{self.layers_block_name}.{i}.{name}'] = (
                        gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device)
                    )
                    gptq[name].free()

            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                additional_layer_inputs = {
                    "attention_mask": layer_attention_mask
                }
                layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                layer_output = move_to_device(
                    layer(layer_input, **additional_layer_inputs)[0],
                    cur_layer_device if cache_examples_on_gpu else CPU
                )
                layer_outputs.append(layer_output)

            layers[i] = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=self.quantize_config.desc_act,
            warmup_triton=autotune_warmup_after_quantized,
            force_layer_back_to_cpu=force_layer_back_to_cpu
        )
        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = simple_dispatch_model(self.model, device_map)
        self.model.config.use_cache = forward_pass_use_cache

        self._quantized = True

        torch.cuda.empty_cache()

    @property
    def device(self):
        if not self.hf_device_map:
            return self.model.device
        else:
            device = [d for d in self.hf_device_map.values() if d not in {'cpu', 'disk'}][0]
            return torch.device(device)

    def to(self, device: Union[str, torch.device]):
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def push_to_hub(
            self,
            repo_id: str,
            save_dir: Optional[str] = None,
            use_safetensors: Optional[bool] = True,
            commit_message: Optional[str] = "Upload of AutoGPTQ quantized model",
            use_auth_token: Optional[Union[bool, str]] = None,
            private: Optional[bool] = None,
            token: Optional[Union[bool, str]] = None,
            create_pr: Optional[bool] = False,
    ) -> str:
        # We don't have an HF model.
        raise NotImplementedError()

    def save_quantized(self, save_dir: str, use_safetensors: bool = False):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if not self.quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")

        self.model.to(CPU)

        model_base_name = self.quantize_config.model_file_base_name or f"gptq_model-{self.quantize_config.bits}bit-{self.quantize_config.group_size}g"
        if use_safetensors:
            model_save_name = model_base_name + ".safetensors"
            state_dict = self.model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            safe_save(state_dict, os.path.join(save_dir, model_save_name))
        else:
            model_save_name = model_base_name + ".bin"
            torch.save(self.model.state_dict(), os.path.join(save_dir, model_save_name))

        self.model.config.save_pretrained(save_dir)
        self.quantize_config.save_pretrained(save_dir)
        self.quantize_config.model_name_or_path = save_dir
        self.quantize_config.model_file_base_name = model_base_name

    def save_pretrained(self, save_dir: str, use_safetensors: bool = False, **kwargs):
        """alias of save_quantized"""
        logger.warning("you are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir, use_safetensors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        **model_init_kwargs
    ):
        """load un-quantized pretrained model to cpu"""

        if not torch.cuda.is_available():
            raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        # TODO: remove need for config.
        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        # if config.model_type not in SUPPORTED_MODELS:
        #     raise TypeError(f"{config.model_type} isn't supported yet.")

        # enforce some values despite user specified

        # TODO: no model_init_kwargs
        # model_init_kwargs["torch_dtype"] = torch_dtype
        # model_init_kwargs["trust_remote_code"] = trust_remote_code
        # if max_memory:
        #     if "disk" in max_memory:
        #         raise NotImplementedError("disk offload not support yet.")
        #     with accelerate.init_empty_weights():
        #         # model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        #         model = Olmo.from_checkpoint(pretrained_model_name_or_path)  # TODO: device?
        #     # TODO: no weight tying?
        #     # model.tie_weights()
        #
        #     max_memory = accelerate.utils.get_balanced_memory(
        #         model,
        #         max_memory=max_memory,
        #         no_split_module_classes=[cls.layer_type],
        #         dtype=model_init_kwargs["torch_dtype"],
        #         low_zero=False
        #     )
        #     model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
        #         model,
        #         max_memory=max_memory,
        #         no_split_module_classes=[cls.layer_type],
        #         dtype=model_init_kwargs["torch_dtype"]
        #     )
        #     model_init_kwargs["low_cpu_mem_usage"] = True
        #
        #     del model
        # else:
        #     model_init_kwargs["device_map"] = None
        #     model_init_kwargs["low_cpu_mem_usage"] = False

        torch.cuda.empty_cache()

        # TODO: do these model_
        model = Olmo.from_checkpoint(pretrained_model_name_or_path)  #, **model_init_kwargs)
        model_config = model.config.asdict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "max_sequence_length"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return cls(model, False, quantize_config)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        use_triton: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        inject_fused_attention: bool = True,
        inject_fused_mlp: bool = True,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = False,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        trainable: bool = False,
        **kwargs
    ):
        """load quantized model from local disk"""

        # Not loading from the hub

        # # Parameters related to loading from Hugging Face Hub
        # cache_dir = kwargs.pop("cache_dir", None)
        # force_download = kwargs.pop("force_download", False)
        # resume_download = kwargs.pop("resume_download", False)
        # proxies = kwargs.pop("proxies", None)
        # local_files_only = kwargs.pop("local_files_only", False)
        # use_auth_token = kwargs.pop("use_auth_token", None)
        # revision = kwargs.pop("revision", None)
        # subfolder = kwargs.pop("subfolder", "")
        # commit_hash = kwargs.pop("_commit_hash", None)

        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        # == step1: prepare configs and file names == #
        if model_name_or_path and save_dir:
            logger.warning("save_dir will be ignored because model_name_or_path is explicit specified.")
        if not model_name_or_path and save_dir:
            model_name_or_path = save_dir
            warnings.warn("save_dir is deprecated and will be removed in version 0.3.0", PendingDeprecationWarning,
                          stacklevel=2)
        if not model_name_or_path and not save_dir:
            raise ValueError("at least one of model_name_or_path or save_dir should be specified.")

        # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        #
        # if config.model_type not in SUPPORTED_MODELS:
        #     raise TypeError(f"{config.model_type} isn't supported yet.")

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, **kwargs)

        if model_basename is None:
            if quantize_config.model_file_base_name:
                model_basename = quantize_config.model_file_base_name
            else:
                model_basename = f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g"

        quantize_config.model_name_or_path = model_name_or_path
        quantize_config.model_file_base_name = model_basename

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)

        resolved_archive_file = None
        if is_local:
            model_save_name = os.path.join(model_name_or_path, model_basename)

            for ext in extensions:
                if os.path.isfile(model_save_name + ext):
                    resolved_archive_file = model_save_name + ext
                    break
        else:  # remote
            # We don't deal with remote files yet.
            pass

        if resolved_archive_file is None:  # Could not find a model file to use
            raise FileNotFoundError(f"Could not find model in {model_name_or_path}")

        model_save_name = resolved_archive_file

        if not use_triton and trainable:
            logger.warning(
                "QuantLinear with cuda backend not support trainable mode yet, Switch to the pytorch backend.")

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip


        # TODO: not doing transformers model
        # transformers.modeling_utils._init_weights = False

        # init_contexts = [no_init_weights()]
        # if low_cpu_mem_usage:
        #     init_contexts.append(accelerate.init_empty_weights(include_buffers=False))

        # with ContextManagers(init_contexts):
            # model = AutoModelForCausalLM.from_config(
            #     config,
            #     trust_remote_code=trust_remote_code,
            #     torch_dtype=torch_dtype
            # )

        # TODO: device
        model = Olmo.from_checkpoint(resolved_archive_file)

        layers = find_layers(model)
        ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                del layers[name]

        make_quant(
            model,
            layers,
            quantize_config.bits,
            quantize_config.group_size,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=quantize_config.desc_act,
            trainable=trainable
        )

        # TODO: no weight tying.
        # model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == "cuda" else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = accelerate.utils.get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type],
                    low_zero=(device_map == "balanced_low_0")
                )
        if not isinstance(device_map, dict):
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type]
            )

        if low_cpu_mem_usage:
            make_sure_no_tensor_in_meta_device(model, use_triton, quantize_config.desc_act, quantize_config.group_size)

        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True
        )
        model = simple_dispatch_model(model, device_map)

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "max_sequence_length"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # == step5: (optional) inject optimized module == #
        if inject_fused_attention:
            if cls.fused_attn_module_type is None:
                inject_fused_attention = False
                logger.warning(f"{cls.__name__} hasn't fused attention module yet, will skip inject fused attention.")
            else:
                cls.fused_attn_module_type.inject_to_model(
                    model,
                    use_triton=use_triton,
                    group_size=quantize_config.group_size,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act,
                    trainable=trainable
                )
        if inject_fused_mlp:
            if cls.fused_mlp_module_type is None:
                inject_fused_mlp = False
                logger.warning(f"{cls.__name__} hasn't fused mlp module yet, will skip inject fused mlp.")
            else:
                cls.fused_mlp_module_type.inject_to_model(
                    model,
                    use_triton=use_triton
                )

        model.eval()
        # == step6: (optional) warmup triton == #

        # TODO: not doing anything for triton yet.
        # if use_triton and warmup_triton:
        #     from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear
        #     QuantLinear.warmup(model, seqlen=model.seqlen)
        #
        #     if inject_fused_mlp and cls.fused_mlp_module_type is not None:
        #         cls.fused_mlp_module_type.warmup(model, seqlen=model.seqlen)

        # == step7: make model compatible with peft
        cls.make_sure_compatible_with_peft(
            model, use_triton, quantize_config.desc_act, quantize_config.group_size
        )

        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton,
            injected_fused_attention=inject_fused_attention,
            injected_fused_mlp=inject_fused_mlp and use_triton,
            trainable=trainable
        )

    # def warmup_triton(self, enabled: bool = True):
    #     if not enabled:
    #         return
    #     if not TRITON_AVAILABLE:
    #         logger.warning(f"triton is not available, skip warmup stage directly.")
    #         return
    #
    #     from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear
    #     QuantLinear.warmup(self.model, seqlen=self.model.seqlen)
    #
    #     if self.fused_mlp_module_type is not None:
    #         self.fused_mlp_module_type.warmup(self.model, seqlen=self.model.seqlen)

    def enable_trainable_mode(self, enabled: bool = True):
        if not self.is_triton_backend and enabled:
            raise NotImplementedError("For now, trainable mode only supports triton backend.")
        for n, m in self.model.named_modules():
            if hasattr(m, "trainable"):
                setattr(m, "trainable", enabled)

    def disable_trainable_mode(self):
        self.enable_trainable_mode(enabled=False)

    @staticmethod
    def make_sure_compatible_with_peft(model: torch.nn.Module, use_triton: bool, desc_act: bool, group_size: int):
        GeneralQuantLinear.inject_to_model(
            model,
            dynamically_import_QuantLinear(use_triton, desc_act, group_size)
        )

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            return getattr(self.model, item)