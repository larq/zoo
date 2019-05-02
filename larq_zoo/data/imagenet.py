from larq_swarm import register_preprocess


@register_preprocess("imagenet2012")
def default(image):
    return image
