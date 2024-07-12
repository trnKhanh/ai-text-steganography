from typing import Union, Callable

import torch


class SeedSchemeFactory:
    registry = {}

    @classmethod
    def register(cls, name: str):
        """
        Register the hash scheme by name. Hash scheme must be callable.

        Args:
            name: name of seed scheme.
        """

        def wrapper(wrapped_class):
            if name in cls.registry:
                print(f"Override {name} in SeedSchemeFactory")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def get_instance(cls, name: str, *args, **kwargs):
        """
        Get the hash scheme by name.

        Args:
            name: name of seed scheme.
        """
        if name in cls.registry:
            return cls.registry[name](*args, **kwargs)
        else:
            return None


class SeedScheme():
    def __call__(self, input_ids: torch.Tensor) -> int:
        return 0

from seed_schemes import *
