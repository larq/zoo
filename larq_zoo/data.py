from larq_flock import registry


@registry.register_preprocess("imagenet2012")
def default(image):
    return image
