import torch
from fastapi import FastAPI

from stegno import generate, decrypt
from utils import load_model

app = FastAPI()


@app.get("/encrypt")
async def encrypt(
    prompt: str,
    msg: str,
    gen_model: str = "openai-community/gpt2",
    device: str = "cpu",
    start_pos: int = 0,
    gamma: float = 2.0,
    msg_base: int = 2,
    seed_scheme: str = "dummy_hash",
    window_length: int = 1,
    private_key: int = 0,
    max_new_tokens_ratio: float = 2,
    num_beams: int = 4,
):
    model, tokenizer = load_model(gen_model, torch.device(device))
    text, msg_rate = generate(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        msg=str.encode(msg),
        start_pos_p=[start_pos],
        gamma=gamma,
        msg_base=msg_base,
        seed_scheme=seed_scheme,
        window_length=window_length,
        private_key=private_key,
        max_new_tokens_ratio=max_new_tokens_ratio,
        num_beams=num_beams,
    )
    return {"text": text, "msg_rate": msg_rate}


@app.get("/decrypt")
async def dec(
    text: str,
    gen_model: str = "openai-community/gpt2",
    device: str = "cpu",
    msg_base: int = 2,
    seed_scheme: str = "dummy_hash",
    window_length: int = 1,
    private_key: int = 0,
):
    model, tokenizer = load_model(gen_model, torch.device(device))
    msgs = decrypt(
        tokenizer=tokenizer,
        device=model.device,
        text=text,
        msg_base=msg_base,
        seed_scheme=seed_scheme,
        window_length=window_length,
        private_key=private_key,
    )
    msg_text = ""
    for i, msg in enumerate(msgs):
        msg_text += f"Shift {i}: {msg}\n\n"
    return {"msg": msg_text}
