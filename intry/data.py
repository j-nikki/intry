from collections import defaultdict
from functools import reduce
from io import BytesIO
from itertools import chain
from pathlib import Path
import pickle
import re
from typing import Dict, List, Set, Tuple
from urllib.request import urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as et

from .utils import *
from .intrin import *
from .tags import *


class data:
    @property
    def wpkey(self) -> int:
        return self._wpkey

    @property
    def wlkey(self) -> int:
        return self._wlkey

    @property
    def wtkey(self) -> int:
        return self._wtkey

    def __init__(self, d: Dict[str, List[intrin]], wp, wl, wt):
        self._dict = d
        self.__getitem__ = self._dict.__getitem__
        self.keys = self._dict.keys
        self.values = self._dict.values
        self.items = self._dict.items
        self._wpkey = wp
        self._wlkey = wl
        self._wtkey = wt


_tses = Tuple[Set[int], int | None, str | None, tags]
_vdata, _vses = 0, 1


def get_data_src(data_source: Path) -> Tuple[str, str, str]:
    if data_source:
        print_(f'importing data from {data_source}...')
        src = data_source
    else:
        url = 'https://cdrdv2.intel.com/v1/dl/getContent/671338'
        print_(f'downloading data from {url}...')
        src = BytesIO(urlopen(url).read())
    with ZipFile(src) as f:
        # TODO: use notes
        def rd(x):
            return f.read(next(y for y in f.filelist if Path(y.filename).name == x)).strip()
        return rd('data.js').lstrip(b'var data_js = "').rstrip(b'";').decode('unicode_escape'), \
            eval(rd('perf.js').decode('U8').lstrip('perf_js =')), \
            eval(re.sub(r'([,{])(\w+):', r'\1"\2":',
                 rd('perf2.js').decode('U8').lstrip('perf2_js =')))


def get_data(data_source: Path) -> data:
    path = user_data_dir('data')
    if os.path.exists(path) and not data_source:
        with open(path, 'rb') as f:
            ver, dat = pickle.load(f)
            if ver == _vdata:
                return dat
    dat: Dict[str, List[intrin]] = defaultdict(list)
    xml, p_, p2 = get_data_src(data_source)
    print_('reading dat...')
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
        dat[i.attrib['tech']].append(intrin(
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
    print_('writing to disk...')
    dat = data(dat, ws[0], ws[1], ws[2])
    with open(path, 'wb') as f:
        pickle.dump((_vdata, dat), f)
    return dat


def get_ses() -> _tses:
    path = user_data_dir('ses')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            ver, ses = pickle.load(f)
            if ver == _vses:
                return ses
    return set(), None, None, tags()


def dump_ses(ses: _tses):
    path = user_data_dir('ses')
    with open(path, 'wb') as f:
        pickle.dump((_vses, ses), f)
