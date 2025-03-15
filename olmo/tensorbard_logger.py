import os.path as osp
from torch.utils.tensorboard import SummaryWriter


# create a new class inheriting from SummaryWriter
class NewSummaryWriter(SummaryWriter):

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


writer = None


def init(log_dir=None):
    global writer
    writer = NewSummaryWriter(log_dir=log_dir)


def log(dictionary, global_step, tag=None):
    global writer
    if writer is None:
        return
    writer.add_scalar_dict(dictionary, global_step, tag)


def write_args_to_tensorboard(args, iteration, prefix=""):
    """Write arguments to tensorboard."""
    global writer
    if writer:
        if prefix:
            prefix = f"{prefix}."
        for arg in args.keys():
            arg_text = f"{prefix}{arg}"
            if isinstance(args[arg], dict):
                write_args_to_tensorboard(args[arg], iteration, prefix=arg_text)
            else:
                writer.add_text(arg_text, str(args[arg]), global_step=iteration)


def finish():
    global writer
    if writer is None:
        return
    writer.close()

