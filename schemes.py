import json
from typing import Literal

from pydantic import BaseModel, Field

from global_config import GlobalConfig
from model_factory import ModelFactory
from seed_scheme_factory import SeedSchemeFactory

with open("resources/examples.json", "r") as f:
    examples = json.load(f)


class EncryptionBody(BaseModel):
    prompt: str | list[str] = Field(title="Prompt used to generate text")
    msg: str = Field(title="Message wanted to hide")
    gen_model: Literal[tuple(ModelFactory.get_models_names())] = Field(
        default=GlobalConfig.get("encrypt.default", "gen_model"),
        title="LLM used to generate text",
    )
    start_pos: int = Field(
        default=GlobalConfig.get("encrypt.default", "start_pos"),
        title="Start position to encrypt the message",
        ge=0,
    )

    delta: float = Field(
        default=GlobalConfig.get("encrypt.default", "delta"),
        title="Hardness parameters",
        gt=0,
    )
    msg_base: int = Field(
        default=GlobalConfig.get("encrypt.default", "msg_base"),
        title="Base of message used in base-encoding",
        ge=2,
    )

    seed_scheme: Literal[tuple(SeedSchemeFactory.get_schemes_name())] = Field(
        default=GlobalConfig.get("encrypt.default", "seed_scheme"),
        title="Scheme used to compute seed for PRF",
    )
    window_length: int = Field(
        default=GlobalConfig.get("encrypt.default", "window_length"),
        title="Window length (context size) used to compute the seed for PRF",
        ge=1,
    )
    private_key: int = Field(
        default=GlobalConfig.get("encrypt.default", "private_key"),
        title="Private key used to compute the seed for PRF",
        ge=0,
    )
    min_new_tokens_ratio: float = Field(
        default=GlobalConfig.get("encrypt.default", "min_new_tokens_ratio"),
        title="Min length of generated text compared to the minimum length required to hide the message",
        ge=1,
    )
    max_new_tokens_ratio: float = Field(
        default=GlobalConfig.get("encrypt.default", "max_new_tokens_ratio"),
        title="Max length of generated text compared to the minimum length required to hide the message",
        ge=1,
    )
    num_beams: int = Field(
        default=GlobalConfig.get("encrypt.default", "num_beams"),
        title="Number of beams used in beam search",
        ge=1,
    )
    do_sample: bool = Field(
        default=GlobalConfig.get("encrypt.default", "do_sample"),
        title="Whether to use greedy or sampling generating"
    )

    repetition_penalty: float = Field(
        default=GlobalConfig.get("encrypt.default", "repetition_penalty"),
        title="Penalty used to avoid repetition when sampling tokens",
        ge=1,
    )
    model_config = {
        "json_schema_extra": {"examples": [examples["encrypt"]["request"]]}
    }


class DecryptionBody(BaseModel):
    text: str = Field(title="Text containing the message")
    gen_model: Literal[tuple(ModelFactory.get_models_names())] = Field(
        default=GlobalConfig.get("decrypt.default", "gen_model"),
        title="LLM used to generate text",
    )
    msg_base: int = Field(
        default=GlobalConfig.get("decrypt.default", "msg_base"),
        title="Base of message used in base-encoding",
        ge=2,
    )
    seed_scheme: Literal[tuple(SeedSchemeFactory.get_schemes_name())] = Field(
        default=GlobalConfig.get("decrypt.default", "seed_scheme"),
        title="Scheme used to compute seed for PRF",
    )
    window_length: int = Field(
        default=GlobalConfig.get("decrypt.default", "window_length"),
        title="Window length (context size) used to compute the seed for PRF",
        ge=1,
    )
    private_key: int = Field(
        default=GlobalConfig.get("decrypt.default", "private_key"),
        title="Private key used to compute the seed for PRF",
        ge=0,
    )
    model_config = {
        "json_schema_extra": {"examples": [examples["decrypt"]["request"]]}
    }
