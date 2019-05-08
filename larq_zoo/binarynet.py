from larq_flock import registry, HParams


@registry.register_model
def binarynet(hparams, dataset):
    pass


@registry.register_hparams(binarynet)
def default():
    return HParams()


def BinaryNet():
    raise NotImplementedError()
