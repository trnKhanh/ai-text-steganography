from inspect import isclass
from pkgutil import iter_modules
from importlib import import_module
import os

pkg_dir = os.path.dirname(__file__)
for _, module_name, _ in iter_modules([pkg_dir]):
    module = import_module(f"{__name__}.{module_name}")

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isclass(attr):
            globals()[attr_name] = attr
