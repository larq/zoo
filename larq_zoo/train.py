from larq_swarm import register_train_function


@register_train_function
def train(
    build_model,
    dataset,
    hparams,
    output_dir,
    epochs,
    initial_epoch,
    pretrain_dir,
    tensorboard,
):
    pass
