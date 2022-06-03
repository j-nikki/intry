from typing import Iterable, Pattern
from .intrin import *
from .data import *


_LF = '\n'


def _intrin2key(in_: intrin):
    return f'{in_.name}\n{in_.instr}\n{_LF.join(map(str.strip, in_.descr.splitlines()))}'.lower()


class listing:
    def __init__(self, dat: data, idx: Iterable[int]):
        self._q = None
        self._qr: Pattern | None = None
        self._y = 0
        self._s2i = reduce(
            dict.__or__, ({_intrin2key(y): y for y in x} for x in dat.values()))
        self._ss = [set(map(_intrin2key, x)) for x in dat.values()]
        self.lst = idx

    @property
    def cur(self) -> intrin | None:
        return self._s2i[self._lq[self._y]] if self._y < len(self._lq) else None

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
            self._l = list(reduce(set.__or__, map(self._ss.__getitem__, idx)))
        else:
            self._l = list(reduce(set.__or__, self._ss))
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

    def submit_query(self):
        self._q = None

    def _runq(self):
        if not self._q:
            self._lq = self._l
        elif self._qr:
            self._lq = list(filter(self._qr.search, self._l))
        self._sety(self._y)
        self._lst = list(map(self._s2i.__getitem__, self._lq))

    def _sety(self, y: int):
        self._y = len(self._lq)-1 if y is max_ else max(
            0, min(len(self._lq)-1, y))
