import logging
import os.path as osp

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# create a new class inheriting from SummaryWriter
class TBNewSummaryWriter(SummaryWriter):

    def __init__(self, log_dir=None, comment="", **kwargs):
        super().__init__(log_dir, comment, **kwargs)


    # create a new function that will take dictionary as input
    # and uses built-in add_scalar() function
    # that function combines all plots into one subgroup by a tag
    def add_scalar_dict(self, dictionary, global_step, tag=None):
        for name, val in dictionary.items():
            if tag is not None:
                name = osp.join(tag, name)
            self.add_scalar(name, val, global_step)


    def log(self, dictionary, global_step, tag=None):
        self.add_scalar_dict(dictionary, global_step, tag)


    def write_args_to_tensorboard(self, args, iteration, prefix=""):
        """Write arguments to tensorboard."""
        if prefix:
            prefix = f"{prefix}."
        for arg in args.keys():
            arg_text = f"{prefix}{arg}"
            if isinstance(args[arg], dict):
                self.write_args_to_tensorboard(args[arg], iteration, prefix=arg_text)
            else:
                self.add_text(arg_text, str(args[arg]), global_step=iteration)
