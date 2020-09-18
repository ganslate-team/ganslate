from pathlib import Path
from omegaconf import OmegaConf
from midaGAN.conf import init_config, InferenceConfig

def build_training_conf():
    cli = OmegaConf.from_cli()
    conf = init_config(cli.pop("config"))
    return OmegaConf.merge(conf, cli)

def build_inference_conf():
    # Load the inference configuration
    cli = OmegaConf.from_cli()
    if cli.config:
        inference_conf = OmegaConf.load(cli.pop("config"))
        inference_conf = OmegaConf.merge(inference_conf, cli)
    else:
        inference_conf = cli
    # Init config to perform type checking and check if there are extra or missing entries
    inference_conf = init_config(inference_conf, InferenceConfig, contains_dataclasses=False)

    # Fetch the config that was used during training of this specific run
    train_conf = Path(inference_conf.checkpoint_dir) / "config.yaml"
    train_conf = OmegaConf.load(str(train_conf))

    # Override exising keys in training config with inference-specific params
    train_conf.load_iter = inference_conf.load_iter
    train_conf.dataset = dict(inference_conf.dataset)
    train_conf.logging.checkpoint_dir = inference_conf.checkpoint_dir
    train_conf.logging.inference_dir = inference_conf.inference_dir
    # Inference defaults
    train_conf.dataset.shuffle = False
    train_conf.gan.is_train = False
    train_conf.batch_size = 1
    
    # TODO: dump inference and training reference configs
    conf = init_config(train_conf)
    return conf
