from loguru import logger
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from ganslate.utils.metrics.train_metrics import TrainingMetrics
from ganslate.nn.utils import get_scheduler
from ganslate.utils import communication, io
from ganslate.utils.builders import build_D, build_G


class BaseGAN(ABC):
    """This class is an abstract base class (ABC) for GAN models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseGAN.__init__(self, conf).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, conf):
        """Initialize the BaseGAN class.
        Parameters:
            conf -- TODO
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseGAN.__init__(self, conf)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """

        self.logger = logger
        self.conf = conf
        self.is_train = self.conf.mode == "train"
        self.device = self._specify_device()
        self.output_dir = conf[conf.mode].output_dir

        self.visuals = {}
        self.metrics = {}
        self.losses = {}
        self.optimizers = {}
        self.networks = {}

    def init_networks(self):      
          
        for name in self.networks.keys():        
            
            # Generator
            if name.startswith('G'):
                # Direction of the generator.
                # 'AB' by default, only bi-directional GANs (e.g. CycleGAN) need
                # generator for 'BA' direction as well.
                direction = 'BA' if name.endswith('_BA') else 'AB'
                self.networks[name] = build_G(self.conf, direction, self.device)            
            
            # Discriminator
            elif name.startswith('D'):
                # Discriminator's domain.
                # 'B' by default, only bi-directional GANs (e.g. CycleGAN) need
                # the 'A' domain discriminator as well.
                domain = 'A' if name.endswith('_A') else 'B'
                self.networks[name] = build_D(self.conf, domain, self.device)

    @abstractmethod
    def init_criterions(self):
        """Initialize criterions (losses)"""

    @abstractmethod
    def init_optimizers(self):
        """Initialize optimizers"""

    def init_metrics(self):
        # Intialize training metrics
        self.training_metrics = TrainingMetrics(self.conf)

    def init_schedulers(self):
        self.schedulers = [get_scheduler(optim, self.conf) for optim in self.optimizers.values()]

    def _specify_device(self):
        if torch.distributed.is_initialized():
            local_rank = communication.get_local_rank()
            return torch.device(f"cuda:{local_rank}")  # distributed GPU training
        elif self.conf[self.conf.mode].cuda:
            return torch.device('cuda:0')  # non-distributed GPU training
        else:
            return torch.device('cpu')

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

    def setup(self):
        """Set up a GAN model. Does the following:
            (1) Initialize its networks, criterions, optimizers, metrics and schedulers
            (2) Converts the networks to mixed precision, if specified
            (3) Loads a checkpoint if continuing training or inferencing
            (4) Applies parallelization to the model if possible
        """
        assert 'G' or 'G_AB' in self.networks.keys(), \
            "The (main) generator has to be named `G` or `G_AB`."

        if self.conf[self.conf.mode].mixed_precision:
            try:
                from apex import amp
            except ModuleNotFoundError:
                sys.exit("\nMixed precision not installed! "
                         "Install Nvidia Apex mixed precision support "
                         "by running `ganslate install-nvidia-apex`'\n")
            # Allow the methods to access AMP that's imported here
            globals()["amp"] = amp

        # Initialize Generators and Discriminators
        self.init_networks()

        if self.is_train:
            # Intialize loss functions (criterions) and optimizers
            self.init_criterions()
            self.init_optimizers()
            self.init_metrics()
            self.init_schedulers()
        else:
            self.eval()
            if len(self.networks.keys()) != 1:  # TODO: any nicer way? look at infer() as well
                raise ValueError(
                    "When inferring there should be only one network initialized - generator.")

        if self.conf[self.conf.mode].mixed_precision:
            self.convert_to_mixed_precision()

        if self.conf[self.conf.mode].checkpointing.load_iter:
            self.load_networks(self.conf[self.conf.mode].checkpointing.load_iter)

        num_devices = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
        if num_devices > 1:
            self.parallelize_networks()

        torch.cuda.empty_cache()

    def backward(self, loss, optimizer, retain_graph=False, loss_id=0):
        """Run backward pass in a regular or mixed precision mode; called by methods <backward_D_basic> and <backward_G>.
        Parameters:
            loss (loss class) -- loss on which to perform backward pass
            optimizer (optimizer or a list of optimizers) -- mixed precision scales the loss with its optimizer
            retain_graph (bool) -- specify if the backward pass should retain the graph
            loss_id (int) -- when used in conjunction with the `num_losses` argument to amp.initialize, 
                             enables Amp to use a different loss scale per loss. 
                             By initializing Amp with `num_losses=1` and setting `loss_id=0` for each loss, 
                             it will use a global scaler for all losses.
        """
        if self.conf[self.conf.mode].mixed_precision:
            with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def parallelize_networks(self):
        """Wrap networks in DistributedDataParallel if using a distributed multi-GPU setup. 
        No parallelization is done in case of single-GPU setup.
        """
        for name in self.networks.keys():
            if torch.distributed.is_initialized():
                # if using batchnorm, broadcast_buffer=True will use batch stats from rank 0,
                # otherwise each process keeps its own stats
                self.networks[name] = DistributedDataParallel(self.networks[name],
                                                              device_ids=[self.device],
                                                              output_device=self.device,
                                                              broadcast_buffers=False)
            elif self.conf[self.conf.mode].cuda and torch.cuda.device_count() > 0:
                message = (
                    "Multi-GPU runs must be launched in distributed mode using `torch.distributed.launch`."
                    " Alternatively, set CUDA_VISIBE_DEVICES=<GPU_ID> to use a single GPU if multiple are present."
                )
                raise RuntimeError(message)

    def convert_to_mixed_precision(self):
        """Initializes Nvidia Apex Mixed Precision
        Parameters:
            opt_level (str) -- specifies Amp's optimization level. Accepted values are
                               "O0", "O1", "O2", and "O3". Check Apex documentation.
            num_losses -- Option to tell Amp in advance how many losses/backward passes you plan to use. 
                          When used in conjunction with the `loss_id` argument to amp.scale_loss, enables Amp to use 
                          a different loss scale per loss/backward pass, which can improve stability.
                          If `num_losses=1`, Amp will still support multiple losses/backward passes, 
                          but use a single global loss scale for all of them.
        """
        opt_level = self.conf[self.conf.mode].opt_level
        networks = list(self.networks.values())  # fetch the networks

        # initialize mixed precision on networks and, if training, on optimizers
        if self.is_train:
            num_losses = len(self.networks)  # each network has its own backward pass
            optimizers = list(self.optimizers.values())
            networks, optimizers = amp.initialize(networks,
                                                  optimizers,
                                                  opt_level=opt_level,
                                                  num_losses=num_losses)
        else:
            networks = amp.initialize(networks, opt_level=opt_level)

        # assigns back the returned mixed-precision networks and optimizers
        self.networks = dict(zip(self.networks, networks))
        if self.is_train:
            self.optimizers = dict(zip(self.optimizers, optimizers))

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every iteration"""
        for scheduler in self.schedulers:
            scheduler.step()

    def save_checkpoint(self, iter_idx):
        """Save all the networks, optimizers and, if used, apex mixed precision's state_dict to the disk.

        Parameters:
            iter_idx (int) -- current iteration; used in the filenames (e.g. 3000.pth)
        """
        checkpoint = {}
        checkpoint_path = Path(self.output_dir) / f"checkpoints/{iter_idx}.pth"
        io.mkdirs(checkpoint_path.parent)

        # add all networks to checkpoint
        for name, net in self.networks.items():
            if isinstance(net, DistributedDataParallel):
                checkpoint[name] = net.module.state_dict()  # e.g. checkpoint["D_A"]
            else:
                checkpoint[name] = net.state_dict()

        # add optimizers to checkpoint
        checkpoint['optimizer_G'] = self.optimizers['G'].state_dict()
        checkpoint['optimizer_D'] = self.optimizers['D'].state_dict()

        # save apex mixed precision
        if self.conf[self.conf.mode].mixed_precision:
            checkpoint['amp'] = amp.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_networks(self, iter_idx):
        """Load all the networks, optimizers and, if used, apex mixed precision's state_dict from the disk.
        Parameters:
            iter_idx (int) -- current iteration; used to specify the filenames (e.g. 3000.pth)
        """
        checkpoint_path = Path(self.output_dir).resolve() / f"checkpoints/{iter_idx}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info(f"Loaded the checkpoint from `{checkpoint_path}`")

        # load networks
        for name in self.networks.keys():
            self.networks[name].load_state_dict(checkpoint[name])

        # load amp state
        # TODO: what about opt_level, does it matter if it's different from before?
        # TODO: what if trained per-loss loss-scale and now using global or vice versa? Just reset it, i.e. ignore the amp state_dict?
        if self.conf[self.conf.mode].mixed_precision:
            if "amp" not in checkpoint:
                self.logger.warning("This checkpoint was not trained using mixed precision.")
            else:
                amp.load_state_dict(checkpoint['amp'])
        else:
            if "amp" in checkpoint:
                self.logger.warning("Loading a model trained with mixed precision "
                                    "without having initiliazed mixed precision")

        # load optimizers
        if self.is_train:
            if self.conf[self.conf.mode].checkpointing.load_optimizers:
                self.logger.info("Optimizers' state_dicts are loaded from the checkpoint.")
                self.optimizers['G'].load_state_dict(checkpoint['optimizer_G'])
                self.optimizers['D'].load_state_dict(checkpoint['optimizer_D'])
            else:
                self.logger.info(
                    "Optimizers' state_dicts were not loaded. Optimizers starting from scratch.")

    def set_requires_grad(self, networks, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            networks (network list) -- a list of networks
            requires_grad (bool)    -- whether the networks require gradients or not
        """
        if not isinstance(networks, list):
            networks = [networks]
        for net in networks:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        for name in self.networks.keys():
            self.networks[name].eval()

    def infer(self, input):
        assert 'G' or 'G_AB' in self.networks.keys(), \
            "The network used for inference is either `G` or `G_AB`."

        generator = 'G' if 'G' in self.networks.keys() else 'G_AB'

        with torch.no_grad():
            return self.networks[generator].forward(input)

    def get_loggable_data(self):
        """Return data that is useful for tracking - learning rates, losses and visuals."""
        learning_rates = {}
        for name, optim in self.optimizers.items():
            learning_rates[f"lr_{name}"] = optim.param_groups[0]['lr']

        return learning_rates, self.losses, self.visuals, self.metrics
