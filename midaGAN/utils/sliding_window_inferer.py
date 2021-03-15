from monai.inferers import SlidingWindowInferer
import torch
from typing import Callable, Any

from loguru import logger


class SlidingWindowInferer(SlidingWindowInferer):

    def __init__(self, *args, **kwargs):
        self.logger = logger
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:

        # Check if roi size and full volume size are not matching
        if len(self.roi_size) != len(inputs.shape[2:]):
            self.logger.debug(
                f"ROI size: {self.roi_size} and input volume: {inputs.shape[2:]} do not match \n"
                "Brodcasting ROI size to match input volume size.")

            # If they do not match and roi_size is 2D add another dimension to roi size
            if len(self.roi_size) == 2:
                self.roi_size = [1, *self.roi_size]
            else:
                raise RuntimeError("Unsupported roi size, cannot broadcast to volume. ")

        return super().__call__(inputs, lambda x: self.network_wrapper(network, x))

    def network_wrapper(self, network, x, *args, **kwargs):
        """
        Wrapper handles cases where inference needs to be done using 
        2D models over 3D volume inputs.

        """
        # If depth dim is 1 in [D, H, W] roi size, then the input is 2D and needs
        # be handled accordingly
        if self.roi_size[0] == 1:
            #  Pass [N, C, H, W] to the model as it is 2D.
            x = x.squeeze(dim=2)
            out = network(x, *args, **kwargs)
            #  Unsqueeze the network output so it is [N, C, D, H, W]
            return out.unsqueeze(dim=2)

        else:
            return network(x, *args, **kwargs)
