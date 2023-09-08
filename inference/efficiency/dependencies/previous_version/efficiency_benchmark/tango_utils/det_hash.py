import collections
import hashlib
import io
from abc import abstractmethod
from typing import Any, MutableMapping, Optional, Type

import base58
import dill

ndarray: Optional[Type]
try:
    from numpy import ndarray
except ModuleNotFoundError:
    ndarray = None

TorchTensor: Optional[Type]
try:
    from torch import Tensor as TorchTensor
except ModuleNotFoundError:
    TorchTensor = None


class CustomDetHash:
    """
    By default, :func:`det_hash()` pickles an object, and returns the hash of the pickled
    representation. Sometimes you want to take control over what goes into
    that hash. In that case, derive from this class and implement :meth:`det_hash_object()`.
    :func:`det_hash()` will pickle the result of this method instead of the object itself.

    If you return ``None``, :func:`det_hash()` falls back to the original behavior and pickles
    the object.
    """

    @abstractmethod
    def det_hash_object(self) -> Any:
        """
        Return an object to use for deterministic hashing instead of ``self``.
        """
        raise NotImplementedError()


class DetHashFromInitParams(CustomDetHash):
    """
    Add this class as a mixin base class to make sure your class's det_hash is derived
    exclusively from the parameters passed to ``__init__()``.
    """

    _det_hash_object: Any

    def __new__(cls, *args, **kwargs):
        super_new = super(DetHashFromInitParams, cls).__new__
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            instance = super_new(cls)
        else:
            instance = super_new(cls, *args, **kwargs)
        instance._det_hash_object = (args, kwargs)
        return instance

    def det_hash_object(self) -> Any:
        """Returns a copy of the parameters that were passed to the class instance's ``__init__()`` method."""
        return self._det_hash_object


class DetHashWithVersion(CustomDetHash):
    """
    Add this class as a mixin base class to make sure your class's det_hash can be modified
    by altering a static ``VERSION`` member of your class.

    Let's say you are working on training a model. Whenever you change code that's part of your experiment,
    you have to change the :attr:`~tango.step.Step.VERSION` of the step that's running that code to tell
    Tango that the step has changed and should be re-run. But if
    you are training your model using Tango's built-in :class:`~tango.integrations.torch.TorchTrainStep`,
    how do you change the version of the step? The answer is, leave the version of the step alone, and
    instead add a :attr:`VERSION` to your model by deriving from this class:

    .. code-block:: Python

        class MyModel(DetHashWithVersion):
            VERSION = "001"

            def __init__(self, ...):
                ...
    """

    VERSION: Optional[str] = None

    def det_hash_object(self) -> Any:
        """
        Returns a tuple of :attr:`~tango.common.det_hash.DetHashWithVersion.VERSION` and this instance itself.
        """
        if self.VERSION is not None:
            return self.VERSION, self
        else:
            return None  # When you return `None` from here, it falls back to just hashing the object itself.


_PICKLE_PROTOCOL = 4


class _DetHashPickler(dill.Pickler):
    def __init__(self, buffer: io.BytesIO):
        super().__init__(buffer, protocol=_PICKLE_PROTOCOL)

        # We keep track of how deeply we are nesting the pickling of an object.
        # If a class returns `self` as part of `det_hash_object()`, it causes an
        # infinite recursion, because we try to pickle the `det_hash_object()`, which
        # contains `self`, which returns a `det_hash_object()`, etc.
        # So we keep track of how many times recursively we are trying to pickle the
        # same object. We only call `det_hash_object()` the first time. We assume that
        # if `det_hash_object()` returns `self` in any way, we want the second time
        # to just pickle the object as normal. `DetHashWithVersion` takes advantage
        # of this ability.
        self.recursively_pickled_ids: MutableMapping[int, int] = collections.Counter()

    def save(self, obj, save_persistent_id=True):
        self.recursively_pickled_ids[id(obj)] += 1
        super().save(obj, save_persistent_id)
        self.recursively_pickled_ids[id(obj)] -= 1

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, CustomDetHash) and self.recursively_pickled_ids[id(obj)] <= 1:
            det_hash_object = obj.det_hash_object()
            if det_hash_object is not None:
                return obj.__class__.__module__, obj.__class__.__qualname__, det_hash_object
            else:
                return None
        elif isinstance(obj, type):
            return obj.__module__, obj.__qualname__
        elif callable(obj):
            if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
                return obj.__module__, obj.__qualname__
            else:
                return None
        elif ndarray is not None and isinstance(obj, ndarray):
            # It's unclear why numpy arrays don't pickle in a consistent way.
            return obj.dumps()
        elif TorchTensor is not None and isinstance(obj, TorchTensor):
            # It's unclear why torch tensors don't pickle in a consistent way.
            import torch

            with io.BytesIO() as buffer:
                torch.save(obj, buffer, pickle_protocol=_PICKLE_PROTOCOL)
                return buffer.getvalue()
        else:
            return None


def det_hash(o: Any) -> str:
    """
    Returns a deterministic hash code of arbitrary Python objects.

    If you want to override how we calculate the deterministic hash, derive from the
    :class:`CustomDetHash` class and implement :meth:`CustomDetHash.det_hash_object()`.
    """
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        pickler = _DetHashPickler(buffer)
        pickler.dump(o)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()
