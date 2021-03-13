from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.utils.metrics.val_test_metrics import ValTestMetrics
from midaGAN.utils import environment
from midaGAN.utils.builders import build_gan, build_loader
from midaGAN.utils.io import decollate
from midaGAN.utils.trackers.validation_testing import ValTestTracker


class BaseValTestEngine(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)

        self.data_loaders = build_loader(self.conf)
        # Val and test modes allow multiple datasets, this handles when it's a single dataset
        if not isinstance(self.data_loaders, dict):
            # No name needed when it's a single dataloader
            self.data_loaders = {"": self.data_loaders}
        self.current_data_loader = None

        self.tracker = ValTestTracker(self.conf)
        self.metricizer = ValTestMetrics(self.conf)

    def run(self, current_idx=""):
        self.logger.info(f'{"Validation" if self.conf.mode == "val" else "Testing"} started.')

        for dataset_name, data_loader in self.data_loaders.items():
            self.current_data_loader = data_loader
            for data in self.current_data_loader:
                # Move elements from data that are visuals
                visuals = {
                    "A": data['A'].to(self.model.device),
                    "fake_B": self.infer(data['A']),
                    "B": data['B'].to(self.model.device)
                }

                # Add masks to visuals if they are provided
                if "masks" in data:
                    visuals.update({"masks": data["masks"]})

                metadata = None
                if "metadata" in data:
                    # After decollate, it is a list of length equal to batch_size,
                    # containing separate metadata for each tensor in the mini-batch
                    metadata = decollate(data["metadata"], batch_size=len(visuals["fake_B"]))

                self.save_generated_image(generated_image=visuals["fake_B"],
                                          metadata=metadata,
                                          idx=current_idx,
                                          dataset_name=dataset_name)

                metrics = self._calculate_metrics(visuals)
                self.tracker.add_sample(visuals, metrics)

            self.tracker.push_samples(current_idx, prefix=dataset_name)

    def save_generated_image(self, generated_image, metadata=None, idx="", dataset_name=""):
        # A dataset object has to have a `save()` method if it
        # wishes to save the outputs in a particular way or format
        save_fn = getattr(self.current_data_loader.dataset, "save", False)

        if save_fn:
            # Tolerates `save` methods with and without `metadata` argument
            def save(tensor, save_dir, metadata=None):
                if metadata is None:
                    save_fn(tensor=tensor, save_dir=save_dir)
                else:
                    save_fn(tensor=tensor, save_dir=save_dir, metadata=metadata)

            # Output dir
            save_dir = "saved/"
            if dataset_name:
                save_dir += f"{dataset_name}/"
            if idx:
                save_dir += f"{idx}/"
            save_dir = self.output_dir / save_dir

            # Loop over the batch and save each tensor
            for batch_idx in range(len(generated_image)):
                tensor = generated_image[batch_idx]
                current_metadata = metadata[batch_idx] if metadata is not None else metadata
                save(tensor=tensor, save_dir=save_dir, metadata=current_metadata)

    def _calculate_metrics(self, visuals):
        # TODO: Decide if cycle metrics also need to be scaled
        pred, target = visuals["fake_B"], visuals["B"]

        # Check if dataset has `denormalize` method defined,
        denormalize = getattr(self.current_data_loader.dataset, "denormalize", False)
        if denormalize:
            pred, target = denormalize(pred), denormalize(target)

        metrics = self.metricizer.get_metrics(pred, target)
        # Update metrics with masked metrics if enabled
        metrics.update(self._get_masked_metrics(pred, target, visuals))
        # Update metrics with cycle metrics if enabled.
        metrics.update(self._get_cycle_metrics(visuals))
        return metrics

    def _get_cycle_metrics(self, visuals):
        """
        Compute cycle metrics from visuals
        """
        cycle_metrics = {}
        if self.conf[self.conf.mode].metrics.cycle_metrics:
            assert 'cycle' in self.model.infer.__code__.co_varnames, \
            "If cycle metrics are enabled, please define behavior of inference"\
            "with a `cycle` flag in the model's `infer()` method"
            rec_A = self.infer(visuals["fake_B"], cycle='B')
            cycle_metrics.update(self.metricizer.get_cycle_metrics(rec_A, visuals["A"]))

        return cycle_metrics

    def _get_masked_metrics(self, pred, target, visuals):
        """
        Compute metrics over masks if they are provided from the dataloader
        """
        mask_metrics = {}

        if "masks" in visuals:
            for label, mask in visuals["masks"].items():
                mask = mask.to(pred.device)
                visuals[label] = 2. * mask - 1
                # Get metrics on masked images
                metrics = {f"{k}_{label}": v \
                    for k,v in self.metricizer.get_metrics(pred, target, mask=mask).items()}
                mask_metrics.update(metrics)
                visuals[label] = 2. * mask - 1

            visuals.pop("masks")

        return mask_metrics


class Validator(BaseValTestEngine):

    def __init__(self, conf, model):
        super().__init__(conf)
        self.model = model

    def _set_mode(self):
        self.conf.mode = 'val'


class Tester(BaseValTestEngine):

    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(self.conf)
        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'test'
