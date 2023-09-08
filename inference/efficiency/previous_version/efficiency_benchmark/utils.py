from collections import abc
from typing import Any, Callable, Sequence


class MappedSequence(abc.Sequence):
    """
    Produces a sequence that applies a function to every element of another sequence.

    This is similar to Python's :func:`map`, but it returns a sequence instead of a :class:`map` object.

    :param fn: the function to apply to every element of the inner sequence. The function should take
               one argument.
    :param inner_sequence: the inner sequence to map over

    From https://github.com/allenai/tango/blob/main/tango/common/sequences.py#L176
    """

    def __init__(self, fn: Callable, inner_sequence: Sequence):
        self.inner = inner_sequence
        self.fn = fn

    def __getitem__(self, item):
        if isinstance(item, slice):
            new_inner = None
            try:
                # special case for a special library
                from datasets import Dataset

                if isinstance(self.inner, Dataset):
                    new_inner = self.inner.select(range(*item.indices(len(self.inner))))
            except ImportError:
                pass
            if new_inner is None:
                new_inner = self.inner[item]
            return MappedSequence(self.fn, new_inner)
        else:
            item = self.inner.__getitem__(item)
            return self.fn(item)

    def __len__(self):
        return self.inner.__len__()

    def __contains__(self, item):
        return any(e == item for e in self)
