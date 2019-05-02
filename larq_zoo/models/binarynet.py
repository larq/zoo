from larq_swarm import register_model, register_hparams, HParams


@register_model
def binarynet(hparams, **kwargs):
    pass


@register_hparams(binarynet)
def default():
    return HParams()


def BinaryNet():
    return binarynet(default())
