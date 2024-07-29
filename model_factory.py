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
            if not torch.cuda.is_available():
                if load_device == "cuda" or run_device == "cuda":
                    print("cuda is not available, use cpu instead")
                    load_device = "cpu"
                    run_device = "cpu"

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
            if cls.__load_model(name) is None:
                return None, None

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

    @classmethod
    def compute_perplexity(cls, model_name, text):
        # This code is copied from https://huggingface.co/docs/transformers/perplexity
        model, tokenizer = cls.load_model(model_name)
        if model is None or tokenizer is None:
            return 0
        device = model.device
        encodings = tokenizer(text, return_tensors="pt").to(device)

        max_length = model.config.n_positions
        stride = max_length//2
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean()).item()
        return ppl

