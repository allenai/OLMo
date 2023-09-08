import bisect
import os
import random
import shutil
from collections import abc
from os import PathLike
from typing import Any, Callable, Iterable, MutableSequence, Optional, Sequence, Union


class ShuffledSequence(abc.Sequence):
    """
    Produces a shuffled view of a sequence, such as a list.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.

    :param inner_sequence: the inner sequence that's being shuffled
    :param indices: Optionally, you can specify a list of indices here. If you don't, we'll just shuffle the
                    inner sequence randomly. If you do specify indices, element ``n`` of the output sequence
                    will be ``inner_sequence[indices[n]]``. This gives you great flexibility. You can repeat
                    elements, leave them out completely, or slice the list. A Python :class:`slice` object is
                    an acceptable input for this parameter, and so are other sequences from this module.

    Example:

    .. testcode::
        :hide:

        import random
        random.seed(42)

    .. testcode::

        from tango.common.sequences import ShuffledSequence
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shuffled_l = ShuffledSequence(l)

        print(shuffled_l[0])
        print(shuffled_l[1])
        print(shuffled_l[2])
        assert len(shuffled_l) == len(l)

    This will print something like the following:

    .. testoutput::

        4
        7
        8
    """

    def __init__(self, inner_sequence: Sequence, indices: Optional[Sequence[int]] = None):
        self.inner = inner_sequence
        self.indices: Sequence[int]
        if indices is None:
            self.indices = list(range(len(inner_sequence)))
            random.shuffle(self.indices)
        else:
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            return self.inner[self.indices[i]]
        else:
            return ShuffledSequence(self.inner, self.indices[i])

    def __contains__(self, item) -> bool:
        for i in self.indices:
            if self.inner[i] == item:
                return True
        return False


class SlicedSequence(ShuffledSequence):
    """
    Produces a sequence that's a slice into another sequence, without copying the elements.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.

    :param inner_sequence: the inner sequence that's being shuffled
    :param s: the :class:`~slice` to slice the input with.

    Example:

    .. testcode::

        from tango.common.sequences import SlicedSequence
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sliced_l = SlicedSequence(l, slice(1, 4))

        print(sliced_l[0])
        print(sliced_l[1])
        print(sliced_l[2])
        assert len(sliced_l) == 3

    This will print the following:

    .. testoutput::

        2
        3
        4

    """

    def __init__(self, inner_sequence: Sequence, s: slice):
        super().__init__(inner_sequence, range(*s.indices(len(inner_sequence))))


class ConcatenatedSequence(abc.Sequence):
    """
    Produces a sequence that's the lazy concatenation of multiple other sequences. It does not copy
    any of the elements of the original sequences.

    This assumes that the inner sequences never change. If they do, the results are undefined.

    :param sequences: the inner sequences to concatenate

    Example:

    .. testcode::

        from tango.common.sequences import ConcatenatedSequence
        l1 = [1, 2, 3]
        l2 = [4, 5]
        l3 = [6]
        cat_l = ConcatenatedSequence(l1, l2, l3)

        assert len(cat_l) == 6
        for i in cat_l:
            print(i)

    This will print the following:

    .. testoutput::

        1
        2
        3
        4
        5
        6
    """

    def __init__(self, *sequences: Sequence):
        self.sequences = sequences
        self.cumulative_sequence_lengths = [0]
        for sequence in sequences:
            self.cumulative_sequence_lengths.append(self.cumulative_sequence_lengths[-1] + len(sequence))

    def __len__(self):
        return self.cumulative_sequence_lengths[-1]

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            if i < 0:
                i += len(self)
            if i < 0 or i >= len(self):
                raise IndexError("list index out of range")
            sequence_index = bisect.bisect_right(self.cumulative_sequence_lengths, i) - 1
            i -= self.cumulative_sequence_lengths[sequence_index]
            return self.sequences[sequence_index][i]
        else:
            return SlicedSequence(self, i)

    def __contains__(self, item) -> bool:
        return any(s.__contains__(item) for s in self.sequences)


class MappedSequence(abc.Sequence):
    """
    Produces a sequence that applies a function to every element of another sequence.

    This is similar to Python's :func:`map`, but it returns a sequence instead of a :class:`map` object.

    :param fn: the function to apply to every element of the inner sequence. The function should take
               one argument.
    :param inner_sequence: the inner sequence to map over

    Example:

    .. testcode::

        from tango.common.sequences import MappedSequence

        def square(x):
            return x * x

        l = [1, 2, 3, 4]
        map_l = MappedSequence(square, l)

        assert len(map_l) == len(l)
        for i in map_l:
            print(i)

    This will print the following:

    .. testoutput::

        1
        4
        9
        16

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


class SqliteSparseSequence(MutableSequence[Any]):
    """
    This is a sparse sequence that pickles elements to a Sqlite database.

    When you read from the sequence, elements are retrieved and unpickled lazily. That means creating/opening
    a sequence is very fast and does not depend on the length of the sequence.

    This is a "sparse sequence" because you can set element ``n`` before you set element ``n-1``:

    .. testcode::
        :hide:

        from tango.common.sequences import SqliteSparseSequence
        import tempfile
        dir = tempfile.TemporaryDirectory()
        from pathlib import Path
        filename = Path(dir.name) / "test.sqlite"

    .. testcode::

        s = SqliteSparseSequence(filename)
        element = "Big number, small database."
        s[2**32] = element
        assert len(s) == 2**32 + 1
        assert s[2**32] == element
        assert s[1000] is None
        s.close()

    .. testcode::
        :hide:

        dir.cleanup()

    You can use a ``SqliteSparseSequence`` from multiple processes at the same time. This is useful, for example,
    if you're filling out a sequence and you are partitioning ranges to processes.

    :param filename: the filename at which to store the data
    :param read_only: Set this to ``True`` if you only want to read.
    """

    def __init__(self, filename: Union[str, PathLike], read_only: bool = False):
        from sqlitedict import SqliteDict

        self.table = SqliteDict(filename, "sparse_sequence", flag="r" if read_only else "c")

    def __del__(self):
        if self.table is not None:
            self.table.close(force=True)
            self.table = None

    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            try:
                return self.table[str(i)]
            except KeyError:
                current_length = len(self)
                if i >= current_length or current_length <= 0:
                    raise IndexError("list index out of range")
                elif i < 0 < current_length:
                    return self.__getitem__(i % current_length)
                else:
                    return None
        elif isinstance(i, slice):
            return SlicedSequence(self, i)
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def __setitem__(self, i: Union[int, slice], value: Any):
        if isinstance(i, int):
            current_length = len(self)
            if i < 0:
                i %= current_length
            self.table[str(i)] = value
            self.table["_len"] = max(i + 1, current_length)
            self.table.commit()
        else:
            raise TypeError(f"list indices must be integers, not {i.__class__.__name__}")

    def __delitem__(self, i: Union[int, slice]):
        current_length = len(self)
        if isinstance(i, int):
            if i < 0:
                i %= current_length
            if i >= current_length:
                raise IndexError("list assignment index out of range")
            for index in range(i + 1, current_length):
                self.table[str(index - 1)] = self.table.get(str(index))
            del self.table[str(current_length - 1)]
            self.table["_len"] = current_length - 1
            self.table.commit()
        elif isinstance(i, slice):
            # This isn't very efficient for continuous slices.
            for index in reversed(range(*i.indices(current_length))):
                del self[index]
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def extend(self, values: Iterable[Any]) -> None:
        current_length = len(self)
        index = -1
        for index, value in enumerate(values):
            self.table[str(index + current_length)] = value
        if index < 0:
            return
        self.table["_len"] = current_length + index + 1
        self.table.commit()

    def insert(self, i: int, value: Any) -> None:
        current_length = len(self)
        for index in reversed(range(i, current_length)):
            self.table[str(index + 1)] = self.table.get(str(index))
        self.table[str(i)] = value
        self.table["_len"] = max(i + 1, current_length + 1)
        self.table.commit()

    def __len__(self) -> int:
        try:
            return self.table["_len"]
        except KeyError:
            return 0

    def clear(self) -> None:
        """
        Clears the entire sequence
        """
        self.table.clear()
        self.table.commit()

    def close(self) -> None:
        """
        Closes the underlying Sqlite table. Do not use this sequence afterwards!
        """
        if self.table is not None:
            self.table.close()
            self.table = None

    def copy_to(self, target: Union[str, PathLike]):
        """
        Make a copy of this sequence at a new location.

        :param target: the location of the copy

        This will attempt to make a hardlink, which is very fast, but only works on Linux and if ``target`` is
        on the same drive. If making a hardlink fails, it falls back to making a regular copy. As a result,
        there is no guarantee whether you will get a hardlink or a copy. If you get a hardlink, future edits
        in the source sequence will also appear in the target sequence. This is why we recommend to not use
        :meth:`copy_to()` until you are done with the sequence. This is not ideal, but it is a compromise we make
        for performance.
        """
        try:
            os.link(self.table.filename, target)
        except OSError as e:
            if e.errno == 18:  # Cross-device link
                shutil.copy(self.table.filename, target)
            else:
                raise
