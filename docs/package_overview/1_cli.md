# Using the command line interface

The command line interface for `ganslate` offers a very simple way to interact with various functionalities.

After installing the package, you can type 
```bash
ganslate
```
in the terminal to explore the various features available in the CLI.


These are the various options available

```text
Usage: ganslate [OPTIONS] COMMAND [ARGS]...

  ganslate - GAN image-to-image translation framework made simple and
  extensible.

Options:
  --help  Show this message and exit.

Commands:
  download-dataset     Download a dataset.
  download-project     Download a ganslate project.
  infer                Do inference with a trained model.
  install-nvidia-apex  Install Nvidia Apex for mixed precision support.
  new-project          Initialize a new project.
  test                 Test a trained model.
  train                Train a model.
```