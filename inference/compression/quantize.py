
import torch
from auto_gptq.modeling._utils import pack_model
from auto_gptq.utils.import_utils import TRITON_AVAILABLE
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

def quantize(
    model,
    examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
    batch_size: int = 1,
    use_triton: bool = False,
    use_cuda_fp16: bool = True,
    autotune_warmup_after_quantized: bool = False,
    cache_examples_on_gpu: bool = True
):

    with torch.inference_mode():
        if model.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")
        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        device_map = model.hf_device_map  # no hf device_map?
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

        class LayerHijacker(nn.Module):
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