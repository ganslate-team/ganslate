import subprocess
import click
from ganslate.engines.utils import init_engine

# Interface
@click.group()
def interface():
    """
    GANSLATE: GAN image-to-image translation framework made simple and extensible. 
    """
    pass

# Train
@interface.command(help="Train a model.")
@click.argument("omegaconf_args", nargs=-1)
def train(omegaconf_args):
    init_engine('train', omegaconf_args).run()

# Test
@interface.command(help="Test a trained model. Requires paired data.")
@click.argument("omegaconf_args", nargs=-1)
def test(omegaconf_args):
    init_engine('test', omegaconf_args).run()

# Infer
@interface.command(help="Do inference with a trained model.")
@click.argument("omegaconf_args", nargs=-1)
def infer(omegaconf_args):
    init_engine('infer', omegaconf_args).run()

# New project
@interface.command(help="Initialize a new project.")
@click.argument("name")
@click.argument("path")
def new_project(name, path):
    print(name, path)

# Download project
@interface.command(help="Download a project.")
@click.argument("name")
@click.argument("path")
def download_project(name, path):
    print(name, path)

# Download dataset
@interface.command(help="Download a dataset.")
@click.argument("name")
@click.argument("path")
def download_dataset(name, path):
    download_script_path = "ganslate/utils/scripts/download_cyclegan_datasets.sh"
    subprocess.call(["bash", download_script_path, name, path])

if __name__ == "__main__":
    interface()