import os
from pathlib import Path
from abc import ABC, abstractmethod
from collections import OrderedDict
import logging

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler

from midaGAN.nn.utils import get_scheduler
from midaGAN.utils import communication


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
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """

        self.logger = logging.getLogger(type(self).__name__)
        self.conf = conf
        self.is_train = conf.gan.is_train
        self.device = self._specify_device()
        self.checkpoint_dir = conf.logging.checkpoint_dir

        self.visuals = {}
        self.losses = {}
        self.optimizers = {}
        self.networks = {}

        self.scaler = GradScaler(enabled=self.conf.mixed_precision)
    
    def _specify_device(self):
        if torch.distributed.is_initialized():
            rank = communication.get_local_rank()
            return torch.device(f"cuda:{rank}")  # distributed GPU training
        elif self.conf.use_cuda: 
            return torch.device('cuda:0')  # non-distributed GPU training
        else:
            return torch.device('cpu')

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self):
        """Final step of initializing a GAN model. Does following:
            (1) Converts the model to mixed precision, if specified
            (2) Sets up schedulers 
            (3) Loads a checkpoint if continuing training or inferencing            
            (4) Applies parallelization to the model if possible
        """        
        if self.is_train:
            self.schedulers = [get_scheduler(optimizer, self.conf) for optimizer in self.optimizers.values()]
        else:
            self.eval()
            if len(self.networks.keys()) != 1: # TODO: any nicer way? look at infer() as well
                raise ValueError("When inferring there should be only one network initialized - generator.")
        
        if self.conf.load_checkpoint:
            self.load_networks(self.conf.load_checkpoint.iter)
        
        num_devices = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
        if num_devices > 1:
            self.parallelize_networks()

        torch.cuda.empty_cache()

    def backward(self, loss, retain_graph=False):
        """Run backward pass in a regular or mixed precision mode; called by methods <backward_D_basic> and <backward_G>.
        Parameters:
            loss (loss class) -- loss on which to perform backward pass
            retain_graph (bool) -- specify if the backward pass should retain the graph
        """
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def optimizer_step(self, optimizer):
        self.scaler.step(self.optimizers[optimizer])

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
            elif self.conf.use_cuda and torch.cuda.device_count() > 0:
                message = ("Multi-GPU runs must be launched in distributed mode using"
                           " `torch.distributed.launch`. Alternatively, set CUDA_VISIBE_DEVICES=<GPU_ID>"
                           " to use a single GPU if multiple are present.")
                raise RuntimeError(message)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every iteration"""
        for scheduler in self.schedulers:
            scheduler.step()

    def save_checkpoint(self, iter_idx):
        """Save all the networks, optimizers and, if used, apex mixed precision's state_dict to the disk.

        Parameters:
            iter_idx (int) -- current iteration; used in the filenames (e.g. 30_net_D_A.pth, 30_optimizers.pth)
        """
        checkpoint = {}
        checkpoint_path = Path(self.checkpoint_dir) / f"{iter_idx}_checkpoint.pth"

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
        if self.conf.mixed_precision:
            checkpoint['grad_scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
    
    def load_networks(self, iter_idx):
        """Load all the networks, optimizers and, if used, apex mixed precision's state_dict from the disk.
        Parameters:
            iter_idx (int) -- current iteration; used to specify the filenames (e.g. 30_net_D_A.pth, 30_optimizers.pth)
        """
        checkpoint_path = Path(self.checkpoint_dir).resolve() / f"{iter_idx}_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info(f"Loaded the checkpoint from `{checkpoint_path}`")
        
        # load networks
        for name in self.networks.keys():
            self.networks[name].load_state_dict(checkpoint[name])

        # load GradScaler state
        if self.conf.mixed_precision:
            if "grad_scaler" not in checkpoint:
                self.logger.warning("This checkpoint was not trained using mixed precision.") 
            else:
                self.scaler.load_state_dict(checkpoint['grad_scaler'])
        else:
            if "grad_scaler" in checkpoint:
                self.logger.warning("Loading a model trained with mixed precision while mixed precision is off.")
        
        # load optimizers   
        if self.is_train:
            if self.conf.load_checkpoint.reset_optimizers:
                self.logger.info("Optimizers' state_dicts were not loaded. Optimizers starting from scratch.")
            else:
                self.logger.info("Optimizers' state_dicts are loaded from the checkpoint.")
                self.optimizers['G'].load_state_dict(checkpoint['optimizer_G']) 
                self.optimizers['D'].load_state_dict(checkpoint['optimizer_D']) 

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
        if self.is_train:
            raise ValueError("Inference cannot be done in training mode.")
        with torch.no_grad():
            generator = list(self.networks.keys())[0] # in inference mode only generator is defined # TODO: any nicer way 
            return self.networks[generator].forward(input)
            
    def get_learning_rates(self):
        """ Return current learning rates of both generator and discriminator"""
        learning_rates = {}
        learning_rates["lr_G"] = self.optimizers['G'].param_groups[0]['lr']
        learning_rates["lr_D"] = self.optimizers['D'].param_groups[0]['lr']
        return learning_rates 

    def get_current_visuals(self):
        return self.visuals
        
    def get_current_losses(self):
        return self.losses

    def get_loggable_data(self):
        """Return data that is useful for tracking - learning rates, losses and visuals."""
        return self.get_learning_rates(), self.get_current_losses(), self.get_current_visuals()