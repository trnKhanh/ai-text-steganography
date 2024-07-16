import base64

import torch
from fastapi import FastAPI
import uvicorn

from stegno import generate, decrypt
from utils import load_model
from seed_scheme_factory import SeedSchemeFactory
from model_factory import ModelFactory
from global_config import GlobalConfig
from schemes import DecryptionBody, EncryptionBody

app = FastAPI()


@app.post("/encrypt")
async def encrypt_api(
    body: EncryptionBody,
):
    model, tokenizer = ModelFactory.load_model(body.gen_model)
    text, msg_rate = generate(
        tokenizer=tokenizer,
        model=model,
        prompt=body.prompt,
        msg=str.encode(body.msg),
        start_pos_p=[body.start_pos],
        gamma=body.gamma,
        msg_base=body.msg_base,
        seed_scheme=body.seed_scheme,
        window_length=body.window_length,
        private_key=body.private_key,
        max_new_tokens_ratio=body.max_new_tokens_ratio,
        num_beams=body.num_beams,
    )
    return {"text": text, "msg_rate": msg_rate}


@app.post("/decrypt")
async def decrypt_api(body: DecryptionBody):
    model, tokenizer = ModelFactory.load_model(body.gen_model)
    msgs = decrypt(
        tokenizer=tokenizer,
        device=model.device,
        text=body.text,
        msg_base=body.msg_base,
        seed_scheme=body.seed_scheme,
        window_length=body.window_length,
        private_key=body.private_key,
    )
    msg_b64 = {}
    for i, s_msg in enumerate(msgs):
        msg_b64[i] = []
        for msg in s_msg:
            msg_b64[i].append(base64.b64encode(msg))
    return msg_b64


@app.get("/configs")
async def default_config():
    configs = {
        "default": {
            "encrypt": {
                "gen_model": GlobalConfig.get("encrypt.default", "gen_model"),
                "start_pos": GlobalConfig.get("encrypt.default", "start_pos"),
                "gamma": GlobalConfig.get("encrypt.default", "gamma"),
                "msg_base": GlobalConfig.get("encrypt.default", "msg_base"),
                "seed_scheme": GlobalConfig.get(
                    "encrypt.default", "seed_scheme"
                ),
                "window_length": GlobalConfig.get(
                    "encrypt.default", "window_length"
                ),
                "private_key": GlobalConfig.get(
                    "encrypt.default", "private_key"
                ),
                "max_new_tokens_ratio": GlobalConfig.get(
                    "encrypt.default", "max_new_tokens_ratio"
                ),
                "num_beams": GlobalConfig.get("encrypt.default", "num_beams"),
            },
            "decrypt": {
                "gen_model": GlobalConfig.get("encrypt.default", "gen_model"),
                "msg_base": GlobalConfig.get("encrypt.default", "msg_base"),
                "seed_scheme": GlobalConfig.get(
                    "encrypt.default", "seed_scheme"
                ),
                "window_length": GlobalConfig.get(
                    "encrypt.default", "window_length"
                ),
                "private_key": GlobalConfig.get(
                    "encrypt.default", "private_key"
                ),
            },
        },
        "seed_schemes": SeedSchemeFactory.get_schemes_name(),
        "models": ModelFactory.get_models_names(),
    }

    return configs


if __name__ == "__main__":
    port = GlobalConfig.get("server", "port")
    if port is None:
        port = 8000
    else:
        port = int(port)
    uvicorn.run("api:app", host="0.0.0.0", port=port, workers=4)
