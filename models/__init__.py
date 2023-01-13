import timm
import functools
import torch.utils.model_zoo as model_zoo

from .mix_transformer import mix_transformer_encoders

encoders = {}
encoders.update(mix_transformer_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)
    return encoder
