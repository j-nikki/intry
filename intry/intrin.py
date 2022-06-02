class intrin:
    def __init__(self, name, tech, cat, instr, descr, params, ret, header, op, perf):
        self._name = name
        self._tech = tech
        self._cat = cat
        self._instr = instr
        self._descr = descr
        self._params = params
        self._ret = ret
        self._header = header
        self._op = op
        self._perf = perf

    @property
    def name(self) -> str:
        return self._name

    @property
    def tech(self) -> str:
        return self._tech

    @property
    def cat(self) -> str:
        return self._cat

    @property
    def instr(self) -> str:
        return self._instr

    @property
    def descr(self) -> str:
        return self._descr

    @property
    def params(self) -> list:
        return self._params

    @property
    def ret(self) -> str:
        return self._ret

    @property
    def header(self) -> str:
        return self._header

    @property
    def op(self) -> str:
        return self._op

    @property
    def perf(self) -> dict:
        return self._perf

    def __lt__(self, other):
        return self.name < other.name
