from typing import Callable, Dict, Generator, Tuple, Type, TypeVar

from .taggers import BaseTagger

T = TypeVar("T", bound=BaseTagger)


class TaggerRegistry:
    __taggers: Dict[str, Type[BaseTagger]] = {}

    @classmethod
    def taggers(cls) -> Generator[Tuple[str, Type[BaseTagger]], None, None]:
        yield from cls.__taggers.items()

    @classmethod
    def add(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        def _add(
            tagger_cls: Type[T],
            tagger_name: str = name,
            taggers_dict: Dict[str, Type[BaseTagger]] = cls.__taggers,
        ) -> Type[T]:
            if tagger_name in taggers_dict and taggers_dict[tagger_name] != tagger_cls:
                raise ValueError(f"Tagger {tagger_name} already exists")

            taggers_dict[tagger_name] = tagger_cls
            return tagger_cls

        return _add

    @classmethod
    def get(cls, name: str) -> Type[BaseTagger]:
        if name not in cls.__taggers:
            raise ValueError(
                f"Unknown tagger {name}; available taggers: " + ", ".join([tn for tn, _ in cls.taggers()])
            )
        return cls.__taggers[name]
