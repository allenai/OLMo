from abc import abstractmethod
from logging import getLogger

import torch.nn as nn
from .triton_utils.mixin import TritonModuleMixin


logger = getLogger(__name__)


class FusedBaseModule(nn.Module, TritonModuleMixin):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs):
        raise NotImplementedError()


class FusedBaseAttentionModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(
        cls,
        model,
        use_triton=False,
        group_size=-1,
        use_cuda_fp16=True,
        desc_act=False,
        trainable=False,
        **kwargs
    ):
        raise NotImplementedError()

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class FusedBaseMLPModule(FusedBaseModule):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        raise NotImplementedError()
