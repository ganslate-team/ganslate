from pathlib import Path
from omegaconf import OmegaConf
from midaGAN.conf import init_config, InferenceConfig, EvalConfig, init_dataclass
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
    train_to_inference_options = ["project_dir", "gan", "generator", 
                                  "use_cuda", "mixed_precision", "opt_level"]

    conf = OmegaConf.masked_copy(conf, train_to_inference_options)

    # Merge conf with inference_defaults and then with cli before init
    conf = OmegaConf.merge(conf, inference_defaults, cli)
    return init_config(conf, InferenceConfig)


def build_eval_conf(conf):
    """
    Evaluation conf works slightly differently from other builders.
    It takes the training conf and modifies it to include evaluation parameters.

    #TODO: Make this more elegant but OmegaConf is a pain while overriding.
    """
    eval_defaults = get_eval_defaults(conf)
    eval_conf = OmegaConf.select(conf, "eval")

    train_to_eval_options = ["project_dir", \
                    "use_cuda", "mixed_precision", "opt_level"]

    conf = OmegaConf.masked_copy(conf, train_to_eval_options)
    eval_conf = OmegaConf.merge(eval_conf, conf, eval_defaults)
    
    try:
        dataset = init_dataclass("dataset", OmegaConf.select(eval_conf, "dataset"))
    
    except ValueError as e:
        # If eval dataset not found in the datasets module, 
        # eval mode will be disabled
        logger.warning(e)
        logger.warning("Evaluation mode turned OFF")
        return None

    OmegaConf.update(eval_conf, "dataset", dataset, merge=True)

    eval_conf = OmegaConf.merge(eval_conf, eval_defaults)

    return eval_conf


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



def get_eval_defaults(conf):
    if "patch_size" in conf.dataset:
        window_size = conf.dataset.patch_size 
    elif "load_size" in conf.dataset:
        # Type will be auto-inferred later
        window_size = str([1, int(conf.dataset.load_size), int(conf.dataset.load_size)])

    eval_defaults = f"""
    batch_size: 1
    dataset:
        name: {"".join(conf.dataset.name.split("Dataset")) + "EvalDataset"}
        shuffle: True
        root: {conf.dataset.root}
        num_workers: 0
            
    sliding_window:
        window_size: {window_size}
    """

    return OmegaConf.create(eval_defaults)    