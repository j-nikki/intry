from curses.ascii import isprint
import fcntl
import os
import re
import termios
import traceback
from argparse import ArgumentParser
from itertools import chain, repeat, islice
from pathlib import Path
from sys import stdin
from tty import setraw
from typing import Dict, Iterable, Generator, List, Tuple

from .utils import *
from .data import *
from .intrin import *
from .listing import *


_ap = ArgumentParser()
_ap.add_argument('ises', type=str, nargs='*', metavar='ISEs')
_ap.add_argument('--data-source', type=Path)
_args = _ap.parse_args()


# =================================


def _main_loop():
    status: None | str = None
    dat = get_data(_args.data_source)
    fs = list(dat.keys())
    wfilts = max(map(len, fs))
    wret = max(len(in_.ret) for in_ in chain.from_iterable(dat.values()))
    fsel, y, sin = get_ses()
    if _args.ises:
        fsel = {fs.index(x) for x in map(str.upper, _args.ises) if x in fs}
        if len(fsel) != len(_args.ises):
            status = '\033[38;5;9munknown ISEs given: ' + \
                ', '.join(set(map(str.upper, _args.ises)) ^ fsel) + '\033[39m'
    x, y, iny = 1, y or 0, 0
    l = listing(dat, fsel)
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
            y = len(fs)-1 if y is max_ else max(0, min(len(fs)-1, y))
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
        b'q': lambda: dump_ses((fsel, y, l.cur and l.cur.name)) or exit(0),
        b'j': lambda: y_(lambda y: y + 1),
        b'k': lambda: y_(lambda y: y - 1),
        b'd': lambda: y_(lambda y: y + 5),
        b'u': lambda: y_(lambda y: y - 5),
        b'G': lambda: y_(lambda _: max_),
        b'g': lambda: y_(lambda _: 0),
        b'h': lambda: x_((x - 1) % 3),
        b'l': lambda: x_((x + 1) % 3),
        b' ': lambda: x == 0 and sel(y),
        b'/': lambda: qry_(''),
    }

    # =================================

    def hl(x):
        return f'\033[22;7m{x}\033[7;27m'

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

    def sig(in_: intrin) -> Generator[ttoken, None, None]:
        yield in_.ret, '\033[38;5;4m'
        yield ' ' + in_.name, '\033[39m'
        yield '(', '\033[38;5;8m'
        for i, (t, n) in enumerate(in_.params):
            if i:
                yield ', ', '\033[38;5;8m'
            yield t + ' ', '\033[38;5;4m'
            yield n, '\033[38;5;14m'
        yield ')', '\033[38;5;8m'

    def infos_impl(in_: intrin, w: int):
        def c(x, pre: str = ''):
            return () if x is None else (pre + ''.join(y) for y in chunk(x, w, ' '))
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
                iny = len(v)-1 if iny is max_ else max(1, min(len(v)-1, iny))
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
        print_(txt, end='')

        s, w_, h_ = gets(1 if l.qry is None else 64, (w_, h_))
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
    print_('\033[?25l\033[s\033[?1049h', end='')
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
        print_('\033[?1049l\033[u\033[?25h', end='')
        if exc:
            print_(f"The program ran into an exception\n{exc.strip()}".replace(
                '\n', '\r\n'))


if __name__ == '__main__':
    main()
