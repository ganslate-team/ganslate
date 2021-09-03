import subprocess
import inspect
import git
import shutil
from pathlib import Path
import click
from cookiecutter.main import cookiecutter
from ganslate.utils.cli import cookiecutter_templates
from ganslate.utils.cli.scripts import download_datasets
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

# First run
@interface.command(help="Fetch resources for the horse2zebra first run")
@click.argument("path", default="./")
def your_first_run(path):
    template = str(COOKIECUTTER_TEMPLATES_DIR / "your_first_run")
    project_path = cookiecutter(template, output_dir=path)
    download_datasets.download("horse2zebra", project_path)
    
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
    download_datasets.download(name, path)

# Install Nvidia Apex
@interface.command(help="Install Nvidia Apex for mixed precision support.")
@click.option(
    "--cpp/--python",
    default=True,
    help=("C++ support is faster and preferred, use Python fallback "
          "only when CUDA is not installed natively.")
)
def install_nvidia_apex(cpp):
    # TODO: Installing with C++ support is a pain due to CUDA installations,
    # waiting for https://github.com/pytorch/pytorch/issues/40497#issuecomment-908685435
    # to switch to PyTorch AMP and get rid of Nvidia Apex

    # Removes the folder if it already exists from a previous, cancelled, try.
    shutil.rmtree("./nvidia-apex-tmp", ignore_errors=True)
    git.Repo.clone_from("https://github.com/NVIDIA/apex", './nvidia-apex-tmp')

    cmd = 'pip install -v --disable-pip-version-check --no-cache-dir'
    if cpp:
        cmd += ' --global-option="--cpp_ext" --global-option="--cuda_ext"'
    cmd += ' ./nvidia-apex-tmp'

    subprocess.run(cmd.split(' '))
    shutil.rmtree("./nvidia-apex-tmp")

if __name__ == "__main__":
    interface()
