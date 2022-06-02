#    .                       s
#   @88>                    :8                   ..
#   %8P      u.    u.      .88       .u    .    @L
#    .     x@88k u@88c.   :888ooo  .d88B :@8c  9888i   .dL
#  .@88u  ^"8888""8888" -*8888888 ="8888f8888r `Y888k:*888.
# ''888E`   8888  888R    8888      4888>'88"    888E  888I
#   888E    8888  888R    8888      4888> '      888E  888I
#   888E    8888  888R    8888      4888>        888E  888I
#   888E    8888  888R   .8888Lu=  .d888L .+     888E  888I
#   888&   "*88*" 8888"  ^%888*    ^"8888*"     x888N><888'
#   R888"    ""   'Y"      'Y"        "Y"        "88"  888
#    ""                                                88F
#                                                     98"
#                                                   ./"
#                                                  ~`

from curses.ascii import isprint
import fcntl
import os
import pickle
import re
import termios
import traceback
import xml.etree.ElementTree as et
from argparse import ArgumentParser
from collections import defaultdict
from functools import reduce
from io import BytesIO
from itertools import chain, repeat, zip_longest, islice
from pathlib import Path
from select import select
from sys import stdin, stdout, platform
from tty import setraw
from typing import Dict, Iterable, Generator, List, Set, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

_ap = ArgumentParser()
_ap.add_argument('ises', type=str, nargs='*', metavar='ISEs')
_ap.add_argument('--data-source', type=Path)
_args = _ap.parse_args()


def _print(s: str, end: str = '\r\n'):
    stdout.buffer.write((s + end).encode('U8'))
    stdout.buffer.flush()


def _gets(n: int, sz: Tuple[int, int]) -> Tuple[str | None, int, int]:
    while True:
        r, _w, _e = select([stdin.buffer.raw], [], [], .05)
        if r:
            return stdin.buffer.raw.read(n), *sz
        sz2 = os.get_terminal_size()
        if sz2 != sz:
            return None, *sz2


def _chunk(it, n, fillvalue=None):
    return zip_longest(*([iter(it)] * n), fillvalue=fillvalue)


def _user_data_dir(file_name: str) -> Path:
    # https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py
    if platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")
    path = Path(os_path) / "intry"
    if not os.path.exists(path):
        os.mkdir(path)
    return path.expanduser() / file_name

# =================================


class _intrin:
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


class _data:
    _dict: Dict[str, List[_intrin]]

    @property
    def wpkey(self) -> int:
        return self._wpkey

    @property
    def wlkey(self) -> int:
        return self._wlkey

    @property
    def wtkey(self) -> int:
        return self._wtkey

    def __init__(self, d: Dict[str, List[_intrin]], wp, wl, wt):
        self._dict = d
        self.__getitem__ = self._dict.__getitem__
        self.keys = self._dict.keys
        self.values = self._dict.values
        self.items = self._dict.items
        self._wpkey = wp
        self._wlkey = wl
        self._wtkey = wt


_tses = Tuple[Set[int], int | None, str | None]
_vdata, _vses = 0, 0

# =================================


def _get_data_src() -> Tuple[str, str, str]:
    if _args.data_source:
        _print(f'importing data from {_args.data_source}...')
        src = _args.data_source
    else:
        url = 'https://cdrdv2.intel.com/v1/dl/getContent/671338'
        _print(f'downloading data from {url}...')
        src = BytesIO(urlopen(url).read())
    with ZipFile(src) as f:
        # TODO: use notes
        def rd(x):
            return f.read(next(y for y in f.filelist if Path(y.filename).name == x)).strip()
        return rd('data.js').lstrip(b'var data_js = "').rstrip(b'";').decode('unicode_escape'), \
            eval(rd('perf.js').decode('U8').lstrip('perf_js =')), \
            eval(re.sub(r'([,{])(\w+):', r'\1"\2":',
                 rd('perf2.js').decode('U8').lstrip('perf2_js =')))


def _get_data() -> _data:
    path = _user_data_dir('data')
    if os.path.exists(path) and not _args.data_source:
        with open(path, 'rb') as f:
            ver, dat = pickle.load(f)
            if ver == _vdata:
                return dat
    dat: Dict[str, List[_intrin]] = defaultdict(list)
    xml, p_, p2 = _get_data_src()
    _print('reading dat...')
    p = {k: reduce(dict.__or__, chain.from_iterable(v.values()))
         for k, v in chain.from_iterable(map(dict.items, p_.values()))}
    for k, v in p2.items():
        p.setdefault(k.split('_')[0], {}).update(reduce(dict.__or__, v))
    ws = [0, len('Cl'), len('CPI')]
    for d in p.values():
        for k, v in d.items():
            d[k] = v.get('l', ''), v.get('t', '')
            ws = list(map(max, zip(ws, map(len, (k, *d[k])))))
    root = et.fromstring(xml)
    for i in root.findall('intrinsic'):
        def orx(x_):
            class x:
                text = None
                attrib = defaultdict(lambda: None)
            return x if x_ is None else x_
        in_ = orx(i.find('instruction')).attrib['name']
        dat[i.attrib['tech']].append(_intrin(
            i.attrib['name'],
            i.attrib['tech'],
            orx(i.find('category')).text,
            (in_ or '').lower(),
            orx(i.find('description')).text,
            [(p.attrib['type'], p.attrib['varname'])
                for p in i.findall('parameter') if p.attrib['type'] != 'void'],
            orx(i.find('return')).attrib['type'],
            orx(i.find('header')).text,
            (orx(i.find('operation')).text or '').strip(),
            p.get((in_ or '').upper(), {})))
    dat = dict(sorted((k, sorted(v)) for k, v in dat.items()))
    _print('writing to disk...')
    dat = _data(dat, ws[0], ws[1], ws[2])
    with open(path, 'wb') as f:
        pickle.dump((_vdata, dat), f)
    return dat


def _get_ses() -> _tses:
    path = _user_data_dir('ses')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            ver, ses = pickle.load(f)
            if ver == _vses:
                return ses
    return set(), None, None


def _dump_ses(ses: _tses):
    path = _user_data_dir('ses')
    with open(path, 'wb') as f:
        pickle.dump((_vses, ses), f)

# =================================


class _max:
    pass


class _listing:
    def __init__(self, dat: _data, idx: Iterable[int]):
        self._q = None
        self._qr = None
        self._y = 0
        self._s2i = reduce(
            dict.__or__, ({f'{y.name}\n{y.instr}': y for y in x} for x in dat.values()))
        self._ss = [{f'{y.name}\n{y.instr}' for y in x} for x in dat.values()]
        self.lst = idx

    @property
    def cur(self) -> _intrin | None:
        return self._s2i[self._lq[self._y]] if self._y < len(self._lq) else None

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, val: int):
        self._sety(val)

    @property
    def lst(self) -> List[_intrin]:
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
        except:
            self._qr = None
        self._runq()

    def submit_query(self):
        self._q = None

    def _runq(self):
        if self._q:
            if self._qr:
                self._lq = list(filter(self._qr.search, self._l))
        else:
            self._lq = self._l
        self._sety(self._y)
        self._lst = list(map(self._s2i.__getitem__, self._lq))

    def _sety(self, y: int):
        self._y = len(self._lq)-1 if y is _max else max(
            0, min(len(self._lq)-1, y))


# =================================


def _main_loop():
    status: None | str = None
    dat = _get_data()
    fs = list(dat.keys())
    wfilts = max(map(len, fs))
    wret = max(len(in_.ret) for in_ in chain.from_iterable(dat.values()))
    fsel, y, sin = _get_ses()
    if _args.ises:
        fsel = {fs.index(x) for x in map(str.upper, _args.ises) if x in fs}
        if len(fsel) != len(_args.ises):
            status = '\033[38;5;9munknown ISEs given: ' + \
                ', '.join(set(map(str.upper, _args.ises)) ^ fsel) + '\033[39m'
    x, y, iny = 1, y or 0, 0
    l = _listing(dat, fsel)
    if sin:
        in_ = next((i for i, in_ in enumerate(l.lst) if in_.name == sin), None)
        if in_ is not None:
            l.y = in_
    if y is None:
        y = sorted(map(fs.index, fsel))[0] if fsel else 0
    inf: Dict[Tuple[str, int], List[str]] = dict()
    ink: Tuple[str, int] | None = None

    # =================================

    def x_(val):
        nonlocal x
        x = min(2, max(0, val))

    def y_(val):
        nonlocal y, iny
        if x == 0:
            y = val(y)
            y = len(fs)-1 if y is _max else y % len(fs)
        elif x == 1:
            l.y = val(l.y)
        else:
            iny = val(iny)

    def sel(i: int | None):
        fsel.remove(i) if i in fsel else fsel.add(i)
        l.lst = fsel

    def qry_(val):
        l.qry = val

    mots = {
        b'q': lambda: _dump_ses((fsel, y, l.cur and l.cur.name)) or exit(0),
        b'j': lambda: y_(lambda y: y + 1),
        b'k': lambda: y_(lambda y: y - 1),
        b'd': lambda: y_(lambda y: y + 5),
        b'u': lambda: y_(lambda y: y - 5),
        b'G': lambda: y_(lambda _: _max),
        b'g': lambda: y_(lambda _: 0),
        b'h': lambda: x_((x - 1) % 3),
        b'l': lambda: x_((x + 1) % 3),
        b' ': lambda: x == 0 and sel(y),
        b'/': lambda: qry_(''),
    }

    # =================================

    def b(x):
        return f'\033[1m{x}\033[22m'

    def hl(x):
        return f'\033[22;7m{x}\033[7;27m'

    def d(x):
        return f'\033[2m{x}\033[22m'

    def trim(x: str, w: int):
        # doesn't count escape seqs into len
        i = n = 0
        gi = x.__getitem__
        while n < w:
            if gi(i) == '\033':
                i += 1
                while gi(i) != 'm':
                    i += 1
                i += 1
            else:
                i += 1
                n += 1
        return x[:i]

    def window(n, i, h) -> int:
        l = min(n, i + h//2)
        return max(0, l - h)

    cols = {
        "MMX": "3",
        "SSE": "2",
        "SSE2": "10",
        "SSE3": "10",
        "SSSE3": "4",
        "SSE4.1": "12",
        "SSE4.2": "14",
        "AVX": "5",
        "AVX2": "13",
        "FMA": "1",
        "AVX_VNNI": "9",
        "AVX-512": "11",
        "KNC": "15",
        "AMX": "15",
        "SVML": "15",
    }

    def techc(c):
        return f'\033[38;5;{cols.get(c, "8")}m'

    # =================================

    syntax = re.compile(
        r'(//.+)|(\b\d+\b|\b0x[a-fA-F0-9]+\b)|((?:\b(?:DEFINE|FOR|TO|to|DO WHILE|OD|CASE|IF|ELSE|FI|ENDFOR|CASE|OF|ESAC|RETURN)\b *)+)|([- !&=~<>+*?:[\](){},]+|\/|\bOR\b|\bAND\b|\bXOR\b)|(\n)', flags=re.MULTILINE)

    ttoken = Tuple[str, str] | None

    def tokenize(s: str, w: int) -> Generator[ttoken, None, None]:
        i = 0
        for it in syntax.finditer(s):
            a, b = it.span()
            if i < a:
                yield s[i:a], '\033[39m'
            if it[1]:
                yield s[a:b], '\033[38;5;2m'
            elif it[2]:
                yield s[a:b], '\033[38;5;12m'
            elif it[3]:
                yield s[a:b], '\033[38;5;13m'
            elif it[4]:
                yield s[a:b], '\033[38;5;8m'
            else:
                yield None
            i = b
        if i != len(s):
            yield s[i:], '\033[39m'

    def parse(tokens: Iterable[ttoken], w: int) -> Generator[str, None, None]:
        l, n = '', 0
        for t_ in tokens:
            if t_ is None:
                yield l + ' ' * (w - n)
                l, n = '', 0
                continue
            t, c = t_
            n += len(t)
            if n > w:
                off = w-n
                yield f'{l}{c}{t[:off]}\033[39m'
                l, n = f'{c}{t[off:]}', len(t[off:])
            else:
                l = f'{l}{c}{t}'
        if l:
            yield l + ' ' * (w - n)

    def sig(in_: _intrin) -> Generator[ttoken, None, None]:
        yield in_.ret, '\033[38;5;4m'
        yield ' ' + in_.name, '\033[39m'
        yield '(', '\033[38;5;8m'
        for i, (t, n) in enumerate(in_.params):
            if i:
                yield ', ', '\033[38;5;8m'
            yield t + ' ', '\033[38;5;4m'
            yield n, '\033[38;5;14m'
        yield ')', '\033[38;5;8m'

    def infos_impl(in_: _intrin, w: int):
        def c(x, pre: str = ''):
            return () if x is None else (pre + ''.join(y) for y in _chunk(x, w, ' '))
        yield from c((in_.tech or "?") + " / " + (in_.cat or "?"), techc(in_.tech))
        if in_.header:
            yield from c(f'#include <{in_.header}>')
        yield from c(in_.instr)
        yield from parse(sig(in_), w)
        if in_.descr:
            for ln in in_.descr.replace('\t', '  ').splitlines():
                yield from c(ln, '\033[39m')
        if in_.perf:
            yield from c(f'╔{"":═>{dat.wpkey}}╤{"Cl":═^{dat.wlkey}}╤{"CPI":═^{dat.wtkey}}╗')
            for k, (l, t) in in_.perf.items():
                yield from c(f'║{k:>{dat.wpkey}}│{l:>{dat.wlkey}}│{t:>{dat.wtkey}}║')
            yield from c(f'╚{"":═>{dat.wpkey}}╧{"":═^{dat.wlkey}}╧{"":═^{dat.wtkey}}╝')
        if in_.op:
            yield from parse(tokenize(in_.op.replace('\t', '  '), w), w)

    # =================================

    def filts():
        for i, k in enumerate(fs):
            res = k.ljust(wfilts)
            res = hl(res) if y == i else res
            res = f'▌{res}' if i in fsel else f' {res}'
            yield techc(k) + res
        yield from repeat(' ' * (wfilts + 1))

    nname, nty, nvar, nspec = len('\033[39m'), len(
        '\033[38;5;4m'), len('\033[38;5;14m'), len('\033[38;5;8m')

    def intrs(w, h):
        for i, in_ in islice(enumerate(l.lst), window(len(l.lst), l.y+2, h), None):
            res = f'\033[38;5;4m{(in_.ret or "void").rjust(wret)} \033[39m{in_.name}\033[38;5;8m(' + \
                "\033[38;5;8m, ".join(f'\033[38;5;4m{f"{t} "}\033[38;5;14m{n}' for t,
                                      n in in_.params) + '\033[38;5;8m)'
            nfx = nty + nname + \
                + (nvar + nty) * len(in_.params) \
                + nspec * (2 + max(0, len(in_.params)-1))
            res = f'{trim(res, w-1)}\033[38;5;8m…' if len(res) - \
                nfx > w else res.ljust(w + nfx)
            yield f'{techc(in_.tech)}{"█" if l.y == i else "▌"}{hl(res) if x >= 1 and l.y == i else res}'
        yield from repeat('\033[38;5;8m▌' + ' ' * w)

    def infos(w: int, h: int) -> Generator[str, None, None]:
        nonlocal iny, ink
        cur = l.cur
        if cur:
            k = (cur.name, w)
            if k not in inf:
                inf[k] = list(infos_impl(cur, w))
            v = inf[k]
            yield hl(v[0]) if x == 2 else v[0]
            if k != ink:
                ink = k
                iny = 1
            else:
                iny = len(v)-1 if iny is _max else max(1, min(len(v)-1, iny))
            yield from islice(v, iny, None)
        yield from repeat(' ' * w)

    # =================================

    w_, h_ = os.get_terminal_size()
    while True:
        w = w_ - 2
        h = h_ - 1
        win = (w - wfilts) // 2
        winf = w - wfilts - win
        txt = '\033[;H' \
            + ''.join(f'{f}{in_}{i}' for _, f, in_, i in zip(range(h), filts(), intrs(win, h), infos(winf, h))) \
            + '\033[39m' \
            + (f'{status}' if status
               else f'{len(l.lst) and (l.y+1)}/{len(l.lst)}' if l.qry is None
               else "\033[38;5;9m" * l.qry_bad + f'/{l.qry}').ljust(w)
        status = None
        _print(txt, end='')

        s, w_, h_ = _gets(1 if l.qry is None else 64, (w_, h_))
        if s is None:
            continue

        if s == b'\033':
            l.qry = None
        elif l.qry is not None:
            if s == b'\x7f':
                l.qry = l.qry[:-1]
            elif s == b'\x0d':
                l.submit_query()
                x = 1
            else:
                try:
                    l.qry += ''.join(filter(isprint, str(s, 'U8')))
                except:
                    pass
        else:
            f = mots.get(s)
            if f:
                f()


def main():
    _print('\033[?25l\033[s\033[?1049h', end='')
    exc: None | str = None
    fd = stdin.fileno()
    old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        setraw(fd)
        _main_loop()
    except Exception:
        exc = traceback.format_exc()
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _print('\033[?1049l\033[u\033[?25h', end='')
        if exc:
            _print(f"The program ran into an exception\n{exc.strip()}".replace(
                '\n', '\r\n'))


if __name__ == '__main__':
    main()


# from .main import main

# if __name__ == '__main__':
#     main()
