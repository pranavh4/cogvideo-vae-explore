from enum import Enum
from typing import Any, cast

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX,
)

MODEL_CONFIG = "THUDM/CogVideoX-2b"


class AutoEncoderModelType(Enum):
    DEFAULT = "default"
    SPATIAL_FIRST = "spatial-first"

    def __str__(self) -> str:
        return self.value


def get_uninitialized_model(model_type: AutoEncoderModelType) -> AutoencoderKLCogVideoX:
    config = cast(
        dict[str, Any],
        AutoencoderKLCogVideoX.load_config(MODEL_CONFIG, subfolder="vae"),
    )

    if model_type == AutoEncoderModelType.DEFAULT:
        return cast(AutoencoderKLCogVideoX, AutoencoderKLCogVideoX.from_config(config))
    elif model_type == AutoEncoderModelType.SPATIAL_FIRST:
        model = cast(AutoencoderKLCogVideoX, AutoencoderKLCogVideoX.from_config(config))
        model.encoder.down_blocks[0].downsamplers[0].compress_time = False
        model.encoder.down_blocks[1].downsamplers[0].compress_time = True
        model.encoder.down_blocks[2].downsamplers[0].compress_time = True

        model.decoder.up_blocks[0].upsamplers[0].compress_time = False
        model.decoder.up_blocks[1].upsamplers[0].compress_time = True
        model.decoder.up_blocks[2].upsamplers[0].compress_time = True

        return model