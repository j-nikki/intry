from itertools import zip_longest
import os
from pathlib import Path
from select import select
from sys import platform, stdin, stdout
from typing import Tuple


class max_:
    pass


def print_(s: str, end: str = '\r\n'):
    stdout.buffer.write((s + end).encode('U8'))
    stdout.buffer.flush()


def gets(n: int, sz: Tuple[int, int]) -> Tuple[str | None, int, int]:
    while True:
        r, _w, _e = select([stdin.buffer.raw], [], [], .05)
        if r:
            return stdin.buffer.raw.read(n), *sz
        sz2 = os.get_terminal_size()
        if sz2 != sz:
            return None, *sz2


def chunk(it, n, fillvalue=None):
    return zip_longest(*([iter(it)] * n), fillvalue=fillvalue)


def user_data_dir(file_name: str) -> Path:
    # https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py
    if platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        os_path = os.getenv('XDG_DATA_HOME') or f'{os.getenv("HOME")}/.local/share'
    path = Path(os_path) / "intry"
    if not os.path.exists(path):
        os.mkdir(path)
    return path.expanduser() / file_name
