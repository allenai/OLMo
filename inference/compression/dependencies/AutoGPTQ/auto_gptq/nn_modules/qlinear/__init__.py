import torch.nn as nn


class GeneralQuantLinear(nn.Linear):
    def __init__(self, quant_linear_module):
        super().__init__(
            in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures, bias=True
        )
        self.infeatures = quant_linear_module.infeatures
        self.outfeatures = quant_linear_module.outfeatures
        self.bits = quant_linear_module.bits
        self.group_size = quant_linear_module.group_size
        self.maxq = quant_linear_module.maxq

        self.weight.requires_grad = False

        self.weight.data = quant_linear_module.qweight
        self.register_buffer("qweight", quant_linear_module.qweight)
        self.bias.data = quant_linear_module.bias

        self.qweight.requires_grad = False
        self.bias.requires_grad = False

        self.register_buffer("qzeros", quant_linear_module.qzeros)
        self.register_buffer("scales", quant_linear_module.scales)
        self.register_buffer("g_idx", quant_linear_module.g_idx)

        if hasattr(quant_linear_module, "wf"):
            self.wf = quant_linear_module.wf
        if hasattr(quant_linear_module, "kernel_switch_threshold"):
            self.kernel_switch_threshold = quant_linear_module.kernel_switch_threshold
        if hasattr(quant_linear_module, "autogptq_cuda_available"):
            self.autogptq_cuda_available = quant_linear_module.autogptq_cuda_available

        self.trainable = quant_linear_module.trainable

        self.forward = quant_linear_module.forward

    @classmethod
    def inject_to_model(cls, model, target_module_type):
        for name, m in model.named_modules():
            if not isinstance(m, target_module_type):
                continue
            new_m = cls(m)
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, new_m)
