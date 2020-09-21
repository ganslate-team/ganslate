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

    # Fetch the config that was used during training of this specific run
    train_conf = Path(inference_conf.logging.checkpoint_dir) / "config.yaml"
    train_conf = OmegaConf.load(str(train_conf))

    # Copy the run-specific options that are important for inference
    train_to_inference_options = ["project_dir", "gan", "generator", 
                                  "use_cuda", "mixed_precision", "opt_level"]
    for key in train_to_inference_options:
        inference_conf[key] = train_conf[key]

    #
    inference_conf = OmegaConf.merge(inference_conf, cli)
    
    # Inference-time defaults
    inference_conf.dataset.shuffle = False
    inference_conf.gan.is_train = False
    inference_conf.batch_size = 1

    return init_config(inference_conf, InferenceConfig)

