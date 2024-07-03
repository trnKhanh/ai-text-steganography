import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_values_per_byte(base: int):
    values_per_byte = 1
    tmp = 255 // base
    while tmp > 0:
        values_per_byte += 1
        tmp = tmp // base
    return values_per_byte


def bytes_to_base(m: bytes, base: int) -> list[int]:
    values_per_byte = get_values_per_byte(base)
    values = []
    for b in m:
        tmp = []
        for i in range(values_per_byte):
            tmp.append(b % base)
            b = b // base
        values.extend(tmp[::-1])

    return values


def base_to_bytes(values: list[int], base: int) -> bytes:
    values_per_byte = get_values_per_byte(base)

    arr = bytearray()

    i = 0
    while i < len(values):
        tmp = 0
        for _ in range(values_per_byte):
            tmp = tmp * base + values[i]
            i += 1
            if i >= len(values):
                break
        arr.append(tmp)

    return bytes(arr)


def load_model(name: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer
