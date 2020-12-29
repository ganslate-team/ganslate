from pathlib import Path
from omegaconf import OmegaConf
from midaGAN.conf import eval_config, init_config, InferenceConfig, EvalConfig, init_dataclass
import logging

logger = logging.getLogger(__name__)


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
    train_to_inference_options = [
        "project_dir", "gan", "generator", "use_cuda", "mixed_precision", "opt_level"
    ]
    conf = OmegaConf.masked_copy(conf, train_to_inference_options)
    # Merge conf with inference_defaults and then with cli before init
    conf = OmegaConf.merge(conf, inference_defaults, cli)
    return init_config(conf, InferenceConfig)


def build_eval_conf(conf):
    """
    #TODO: Make this more elegant but OmegaConf is a pain while overriding.
    """
    eval_conf = OmegaConf.masked_copy(conf, "evaluation")["evaluation"]
    eval_conf = override_defaults(conf, eval_conf)
    dataset = OmegaConf.to_container(eval_conf.dataset)

    try:
        eval_conf.dataset = init_dataclass("dataset", OmegaConf.select(eval_conf, "dataset"))
    except ValueError as e:
        logger.warning(f"Evaluation mode turned OFF: {e}")
        return None

    for key in dataset:
        eval_conf.dataset[key] = dataset[key]

    return eval_conf


def get_inference_defaults(conf):
    inference_defaults = f"""
    dataset: 
        shuffle: False

    logging:
        checkpoint_dir: {conf.logging.checkpoint_dir}

    """
    return OmegaConf.create(inference_defaults)


def override_defaults(conf, eval_conf):
    tensorboard_config = conf.logging.tensorboard

    if OmegaConf.select(conf, "dataset.patch_size"):
        window_size = conf.dataset.patch_size
    elif OmegaConf.select(conf, "dataset.load_size"):
        # Type will be auto-inferred later
        window_size = str([1, int(conf.dataset.load_size), int(conf.dataset.load_size)])

    eval_defaults = OmegaConf.create(f"""
    logging:
        inference_dir: {conf.logging.checkpoint_dir}
        tensorboard: {tensorboard_config}

    dataset:
        shuffle: True
        num_workers: 0
            
    sliding_window:
        window_size: {window_size}
    """)

    OmegaConf.update(eval_defaults, "logging.wandb", conf.logging.wandb, merge=True)
    return OmegaConf.merge(eval_conf, eval_defaults)
