from contextlib import suppress
from typing import Iterable, List, Callable

from .intrin import intrin

_stags = ' ¹²³⁴⁵⁶⁷⁸⁹'


class tags:
    def __getitem__(self, idx: int) -> str:
        return self._tags[len(self._tags) - idx]

    @property
    def lst(self) -> List[str]:
        return self._tags

    def str_(self, iname: str) -> str:
        if iname not in self._tags:
            return ' '
        idx = len(self._tags) - self._tags.index(iname)
        return _stags[idx] if idx < 10 else '°'

    def push(self, tag: str):
        if tag in self._tags:
            self._tags.remove(tag)
        self._tags.append(tag)

    def pop(self, tag: str) -> str | None:
        with suppress(ValueError):
            self._tags.remove(tag)
            return tag
        with suppress(IndexError):
            return self._tags.pop()

    def __init__(self, tags_: List[str] = []):
        self._tags = tags_
