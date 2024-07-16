from pydantic import BaseModel
from global_config import GlobalConfig


class EncryptionBody(BaseModel):
    prompt: str
    msg: str
    gen_model: str = GlobalConfig.get("encrypt.default", "gen_model")
    start_pos: int = GlobalConfig.get("encrypt.default", "start_pos")

    gamma: float = GlobalConfig.get("encrypt.default", "gamma")
    msg_base: int = GlobalConfig.get("encrypt.default", "msg_base")

    seed_scheme: str = GlobalConfig.get("encrypt.default", "seed_scheme")
    window_length: int = GlobalConfig.get(
        "encrypt.default", "window_length"
    )
    private_key: int = GlobalConfig.get("encrypt.default", "private_key")
    max_new_tokens_ratio: float = GlobalConfig.get(
        "encrypt.default", "max_new_tokens_ratio"
    )
    num_beams: int = GlobalConfig.get("encrypt.default", "num_beams")
    repetition_penalty: float = GlobalConfig.get('encrypt.default', "repetition_penalty")

class DecryptionBody(BaseModel):
    text: str
    gen_model: str = GlobalConfig.get("decrypt.default", "gen_model")
    msg_base: int = GlobalConfig.get("decrypt.default", "msg_base")

    seed_scheme: str = GlobalConfig.get("decrypt.default", "seed_scheme")
    window_length: int = GlobalConfig.get(
        "decrypt.default", "window_length"
    )
    private_key: int = GlobalConfig.get("decrypt.default", "private_key")
