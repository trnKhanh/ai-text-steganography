import torch
import transformers
from utils import load_model, static_init
from global_config import GlobalConfig


@static_init
class ModelFactory:
    models_names = {}
    models = {}
    tokenizers = {}
    run_model = None
    dtype = torch.bfloat16
    load_device = torch.device("cpu")
    run_device = torch.device("cpu")

    @classmethod
    def __static_init__(cls):
        names_sec = GlobalConfig.get_section("models.names")
        if names_sec is not None:
            for name in names_sec:
                cls.models_names[name] = GlobalConfig.get("models.names", name)

        if GlobalConfig.get_section("models.params") is not None:
            dtype = GlobalConfig.get("models.params", "dtype")
            if dtype == "bfloat16":
                cls.dtype = torch.bfloat16
            elif dtype == "float16":
                cls.dtype = torch.float16
            elif dtype == "float32":
                cls.dtype = torch.float32

            load_device = GlobalConfig.get("models.params", "load_device")
            run_device = GlobalConfig.get("models.params", "run_device")
            if load_device is not None:
                cls.load_device = torch.device(str(load_device))
            if run_device is not None:
                cls.run_device = torch.device(str(run_device))

    @classmethod
    def __load_model(cls, name):
        if name not in cls.models_names:
            print(f"{name} is not a valid model name")
            return None

        if name not in cls.models:
            model, tokenizer = load_model(
                cls.models_names[name], cls.load_device
            )
            cls.models[name] = model
            cls.tokenizers[name] = tokenizer
        else:
            model, tokenizer = cls.models[name], cls.tokenizers[name]

        return model, tokenizer

    @classmethod
    def load_model(cls, name):
        if name not in cls.models:
            cls.__load_model(name)

        if name != cls.run_model and cls.run_model is not None:
            cls.models[cls.run_model].to(cls.load_device)

        cls.models[name].to(cls.run_device)
        cls.run_model = name

        return cls.models[name], cls.tokenizers[name]

    @classmethod
    def get_models_names(cls):
        return list(cls.models_names.keys())

    @classmethod
    def get_model_max_length(cls, name: str):
        if name in cls.tokenizers:
            return cls.tokenizers[name].model_max_length
        else:
            return 0
