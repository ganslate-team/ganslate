# Engines

`ganslate` defines four _engines_ that implement processes crucial to deep learning workflow. These are `Trainer`, `Validator`, `Tester`, and `Inferer`. The following UML diagram shows the design of the `ganslate`'s `engines` module and the relationship between the different engine classes defined in it. 

![alt text](../imgs/uml-ganslate_engines.png "Relationship between ganslate's engine classes")



------------
## `Trainer`

The `Trainer` class ([source](https://github.com/ganslate-team/ganslate/ganslate/engines/trainer.py)) implements the training procedure and is instantiated at the start of the training process. Upon initialization, the trainer object executes the following tasks:
1. Preparing the environment.
2. Initializing the GAN model, training data loader, traning tracker, and validator.

The `Trainer` class provides the `run()` method which defines the training logic. This includes:
1. Fetching data from the training dataloader
2. Invoking the GAN model's methods that set the inputs and perform forward pass, backpropagation, and parameter update. 
3. Obtaining the results of the iteration which includes the computed images, loss values, metrics, and I/O and computation times, and pushing them into the experiment tracker for logging.
4. Running model validation.
5. Saving checkpoints locally.
6. Updating the learning rates.

All configuration pertaining to the `Trainer` is grouped under the `'train'` mode in `ganslate`.



--------------
## `Validator`
The `Validator`class ([source](https://github.com/ganslate-team/ganslate/ganslate/engines/validator_tester.py)) inherits almost all of its properties and functionalities from the `BaseValTestEngine`, and is responsible for performing validation given a model during the training process. It is instantiated and utilized within the `Trainer` where it is supplied with its configuration and the model. Upon initialization, a `Validator` object executes the following:
1. Initializes the sliding window inferer, validation data loader, validation tracker, and the validation-test metricizer

The `run()` method of the `Validator` iterates over the validation dataset and executes the following steps:
1. Fetching data from the validation data loader.
2. Running inference on the given model and holding the computed images.
3. Saving the computed image and its relevant metadata (useful in case of medical images).
4. Calculate image quality/similarity metrics by comparing the generated image with the geound truth.
5. Pushing the images and metrics into the validation tracker for logging.

All configuration pertaining to the `Validator` is grouped under the `'val'` mode in `ganslate`.



-----------
## `Tester`
The `Tester` class ([source](https://github.com/ganslate-team/ganslate/ganslate/engines/validator_tester.py)), like the `Validator`, inherits from the `BaseValTestEngine` and has the same properties and functionalities as the `Validator`. The only difference is that a `Tester` instance sets up the environment and builds its own GAN model, and is therefore used independently of the `Trainer`.

All configuration pertaining to the `Tester` is grouped under the `'test'` mode in `ganslate`.



------------
## `Inferer`
The `Inferer` class ([source](https://github.com/ganslate-team/ganslate/ganslate/engines/validator_tester.py)) represents a simplified inference engine without any mechanism for metric calculation. Therefore, it expects data without a ground truth to compare against. It does execute utility tasks like fetching data from a data loader, tracking I/O and computation time, and logging and saving images under normal circumstances. However, when used in the _deployment_ mode, the `Inferer` essentially acts as a minimal inference engine that can be easily integrated into other applications.