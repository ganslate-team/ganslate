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
    conf = OmegaConf.load(cli.pop("config"))
    
    # Inference-time defaults
    inference_defaults = get_inference_defaults(conf)

    # Copy the run-specific options that are important for inference
    train_to_inference_options = ["project_dir", "gan", "generator", 
                                  "use_cuda", "mixed_precision", "opt_level"]

    conf = OmegaConf.masked_copy(conf, train_to_inference_options)

    # Merge conf with inference_defaults and then with cli before init
    conf = OmegaConf.merge(conf, inference_defaults, cli)
    return init_config(conf, InferenceConfig)



def get_inference_defaults(conf):
    inference_defaults = f"""
    batch_size: 1
    dataset: 
        shuffle: False

    gan:
        is_train: False
    
    logging:
        checkpoint_dir: {conf.logging.checkpoint_dir}

    """

    return OmegaConf.create(inference_defaults)