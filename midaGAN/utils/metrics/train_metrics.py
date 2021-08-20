import ganslate.nn.losses.utils.ssim as ssim
import torch


class TrainingMetrics:

    def __init__(self, conf):
        self.output_distributions = True if conf.train.metrics.discriminator_evolution else False

        if conf.train.metrics.ssim:
            self.ssim = ssim.SSIMLoss()
        else:
            self.ssim = None

    def get_output_metric_D(self, out):
        """
        Store fake and real discriminator outputs to analyze training convergence:
        Based on ADA-StyleGAN observations: 
        https://medium.com/swlh/training-gans-with-limited-data-22a7c8ffce78
        """
        if self.output_distributions:
            # Reduce the output to a tensor if it is dict
            if isinstance(out, dict):
                out = torch.tensor([elem.detach().mean() for elem in out.values()])

            else:
                out = out.detach()
                if len(out.size()) > 1:
                    return out.mean()
                else:
                    return out
        else:
            return None

    def get_SSIM_metric(self, input, target):
        # Gradient computation not needed for metric computation
        input = input.detach()
        target = target.detach()
        # Data range needs to be positive and normalized
        # https://github.com/VainF/pytorch-msssim#2-normalized-input
        input = (input + 1) / 2
        target = (target + 1) / 2

        if self.ssim:
            return 1 - self.ssim(input, target, data_range=1)
        else:
            return None

    def compute_metrics_D(self, discriminator, pred_real, pred_fake):
        # Update metrics with output distributions if enabled
        return {
            f"{discriminator}_real": self.get_output_metric_D(pred_real),
            f"{discriminator}_fake": self.get_output_metric_D(pred_fake)
        }

    def compute_metrics_G(self, visuals):
        # Update metrics with SSIM if enabled in config
        metrics_G = {}
        if all([key in visuals for key in ["rec_A", "real_A"]]):
            # Update SSIM for forward A->B->A reconstruction
            metrics_G['ssim_A'] = self.get_SSIM_metric(visuals["real_A"], visuals["rec_A"])

        if all([key in visuals for key in ["rec_B", "real_B"]]):
            # Update SSIM for forward B->A->B reconstruction
            metrics_G['ssim_B'] = self.get_SSIM_metric(visuals["real_B"], visuals["rec_B"])

        return metrics_G
