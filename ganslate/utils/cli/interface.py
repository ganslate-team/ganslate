import subprocess
import inspect
from pathlib import Path
import click
from cookiecutter.main import cookiecutter
from ganslate.utils.cli import cookiecutter_templates
from ganslate.engines.utils import init_engine


COOKIECUTTER_TEMPLATES_DIR = Path(inspect.getfile(cookiecutter_templates)).parent

# Interface
@click.group()
def interface():
    """ganslate - GAN image-to-image translation framework made simple and extensible."""
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
@click.argument("path", default="./")
def new_project(path):
    template = str(COOKIECUTTER_TEMPLATES_DIR / "new_project")
    cookiecutter(template, output_dir=path)

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

# Install Nvidia Apex
@interface.command(help="Install Nvidia Apex for mixed precision support.")
@click.option(
    "--cpp/--python",
    default=True,
    help=("C++ support is faster and preferred, use Python fallback "
          "only when CUDA is not installed natively.")
)
def install_nvidia_apex(cpp):
    # TODO: (Ibro) I need to verify this in a few days when I have access to the GPU
    cmd = 'pip install -v --disable-pip-version-check --no-cache-dir'
    if cpp:
        cmd += ' --global-option="--cpp_ext" --global-option="--cuda_ext"'
    cmd += ' git+https://github.com/NVIDIA/apex.git'
    subprocess.run(cmd.split(' '))

if __name__ == "__main__":
    interface()