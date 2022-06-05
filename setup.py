from setuptools import setup
from Cython.Build import cythonize
from intry import __version__

setup(
    name='intry',
    version=__version__,
    description='Command-line program for browsing intrinsics.',
    python_requires='>=3.10',
    ext_modules=cythonize('intry/*.py'),
    packages=['intry'],
    scripts=['bin/intry'],
    install_requires=['pyperclip'],
)
