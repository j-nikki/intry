from bisect import bisect
from contextlib import suppress
from typing import Iterable, Pattern

from .data import *
from .intrin import *

_LF = '\n'


def _intrin2key(in_: intrin):
    return f'{in_.name}\n{in_.instr}\n{_LF.join(map(str.strip, in_.descr.splitlines()))}'.lower()


class listing:
    def __init__(self, dat: data, idx: Iterable[int]):
        self._q = None
        self._qr: Pattern | None = None
        self._y = 0
        self._s2i = reduce(
            dict.__or__, ({y.name.lower(): y for y in x} for x in dat.values()))
        self._ss = [set(map(_intrin2key, x)) for x in dat.values()]
        self.lst = idx

    @property
    def cur(self) -> intrin | None:
        with suppress(IndexError):
            return self._s2i[self._lq[self._y].split('\n', 1)[0]]

    @cur.setter
    def cur(self, val: str):
        try:
            self._y = next(i for i, x in enumerate(self._lst)
                           if val == x.name.split('\n', 1)[0])
        except StopIteration:
            in_ = self._s2i[val]
            self._y = bisect(self._lst, in_)
            self._lst.insert(self._y, in_)

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, val: int):
        self._sety(val)

    @property
    def lst(self) -> List[intrin]:
        return self._lst

    @lst.setter
    def lst(self, idx: Iterable[int]):
        if idx:
            self._l = list(
                sorted(reduce(set.__or__, map(self._ss.__getitem__, idx))))
        else:
            self._l = list(sorted(reduce(set.__or__, self._ss)))
        self._runq()

    @property
    def qry(self) -> str | None:
        return self._q

    @property
    def qry_bad(self) -> bool:
        return self._qr is None

    @qry.setter
    def qry(self, q: str | None):
        self._q = q
        try:
            self._qr = re.compile(q, flags=re.MULTILINE)
            if self._qr.pattern != q:
                self._q = None
        except:
            self._qr = None
        self._runq()
        
    @property
    def qry_re(self) -> re.Pattern | None:
        return self._qr

    def submit_query(self):
        self._q = None

    def _runq(self):
        if not self._q:
            self._lq = self._l
        elif self._qr:
            self._lq = list(filter(self._qr.search, self._l))
        self._lst = [self._s2i[x.split('\n', 1)[0]] for x in self._lq]
        self._sety(self._y)

    def _sety(self, y: int):
        self._y = len(self._lst)-1 if y is max_ else max(
            0, min(len(self._lst)-1, y))
