# Projects in `ganslate`

In `ganslate`, a _project_ refers to a collection of all custom code and configuration files pertaining to your specific task. The project directory is expected to have a certain structure that isolates logically different parts of the project, such as data pipeline, GAN implementation, and configuration. The directory structure is as follows

```text
<your_project_dir>
    |
    |- datasets
    |   |- custom_train_dataset.py
    |   |- custom_val_test_dataset.py
    |
    |- architectures
    |   |- custom_gan.py
    |   
    |- experiments
    |    |- exp1_config.yaml
    |
    |- __init__.py
    |
    |- README.md
```

The `__init__.py` file initializes your project directory as Python module which is necessary for `ganslate`'s configuration system to correctly function. (See [configuration](./7_configuration.md) for details). The `README.md` file could contain a description of your task. 

`ganslate` provides a Cookiecutter template which can automatically generate an empty project for you. The tutorial [Your First Project](../tutorials_basic/2_new_project.md) provides detailed instructions on how to create and operate your own project.