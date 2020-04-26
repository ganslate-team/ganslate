import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from apex import amp

from . import networks3d as networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.is_train = opt.is_train
        self.device = torch.device('cuda:0') if opt.use_cuda else torch.device('cpu')
        self.num_devices = torch.cuda.device_count()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name) # TODO: conf in folder out
        
        torch.backends.cudnn.benchmark = True

        self.visuals = {}
        self.losses = {}
        self.optimizers = {}
        self.networks = {}
    
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
            (1) Sets up schedulers 
            (2) Converts the model to mixed precision, if specified
            (3) Loads a checkpoint if continuing training or inferencing            
            (4) Applies parallelization to the model if possible
        """
        if self.is_train:
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers.values()]

        if self.opt.mixed_precision:
            self.convert_to_mixed_precision(self.opt.opt_level, self.opt.per_loss_scale)

        if not self.is_train or self.opt.continue_train:
            self.load_networks(self.opt.load_epoch)

        if self.num_devices > 1:
            self.parallelize_networks()
        
        torch.cuda.empty_cache()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.networks.keys():
            self.networks[name].eval()

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()

    def backward(self, loss, optimizer, retain_graph=False, loss_id=0):
        """Run backward pass in a regular or mixed precision mode; called by methods <backward_D_basic> and <backward_G>.
        Parameters:
            loss (loss class) -- loss on which to perform backward pass
            optimizer (optimizer class) -- mixed precision scales the loss with its optimizer
            retain_graph (bool) -- specify if the backward pass should retain the graph
            loss_id (int) -- when used in conjunction with the `num_losses` argument to amp.initialize, 
                             enables Amp to use a different loss scale per loss. 
                             By initializing Amp with `num_losses=1` and setting `loss_id=0` for each loss, 
                             it will use a global scaler for all losses.
        """
        if self.opt.mixed_precision:
            with amp.scale_loss(loss, optimizer, loss_id) as loss_scaled:
                loss_scaled.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def parallelize_networks(self):
        """Wrap networks in DataParallel in case of multi-GPU setup, or in DistributedDataParallel if
        using a distributed setup. No parallelization is done in case of single-GPU setup.
        """
        for name in self.networks.keys():
            if self.opt.distributed:
                self.networks[name] = DistributedDataParallel(self.networks[name],
                                                              device_ids=[self.device], 
                                                              output_device=self.device)
            else:
                self.networks[name] = DataParallel(self.networks[name])

    def convert_to_mixed_precision(self, opt_level='O1', per_loss_scale=False):
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
        networks = list(self.networks.values())

        # initialize mixed precision on networks and, if training, on optimizers
        if self.is_train:
            # there is always one loss/backward pass per network  # TODO: good enough?
            num_losses = len(self.networks) if per_loss_scale else 1 
            optimizers = list(self.optimizers.values())
            networks, optimizers = amp.initialize(networks, optimizers, 
                                                  opt_level=opt_level, num_losses=num_losses)
        else:
            networks = amp.initialize(networks, opt_level=opt_level)

        # assigns back the returned mixed-precision networks and optimizers # TODO: say why
        self.networks = dict(zip(self.networks, networks))
        if self.is_train:
            self.optimizers = dict(zip(self.optimizers, optimizers))

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        """ Return current learning rates of both generator and discriminator"""
        lr_G = self.optimizers['G'].param_groups[0]['lr']
        lr_D = self.optimizers['D'].param_groups[0]['lr']
        return lr_G, lr_D    

    def get_current_visuals(self):
        """Return visualization images. train.py will save the images to a HTML and Weights&Biases"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
        
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks, optimizers and, if used, apex mixed precision's state_dict to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the filenames (e.g. 30_net_D_A.pth, 30_optimizers.pth)
        """
        checkpoint = {}
        checkpoint_path = os.path.join(self.save_dir, '%s_checkpoint.pth' % epoch)

        # add all networks to checkpoint
        for name, net in self.networks.items():
            if isinstance(net, DataParallel) or isinstance(net, DistributedDataParallel):
                checkpoint[name] = net.module.state_dict()  # e.g. checkpoint["D_A"]
            else:
                checkpoint[name] = net.state_dict()

        # add optimizers to checkpoint
        checkpoint['optimizer_G'] = self.optimizers['G'].state_dict()
        checkpoint['optimizer_D'] = self.optimizers['D'].state_dict()

        # save apex mixed precision
        if self.opt.mixed_precision:
            checkpoint['amp'] = amp.state_dict()

        torch.save(checkpoint, checkpoint_path)
    
    def load_networks(self, epoch):
        """Load all the networks, optimizers and, if used, apex mixed precision's state_dict from the disk.
        Parameters:
            epoch (int) -- current epoch; used to specify the filenames (e.g. 30_net_D_A.pth, 30_optimizers.pth)
        """
        checkpoint_path = os.path.join(self.save_dir, '%s_checkpoint.pth' % epoch)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print('loaded the checkpoint from %s' % checkpoint_path) # TODO: make nice logging

        # load networks
        for name in self.networks.keys():
            self.networks[name].load_state_dict(checkpoint[name])

        # load amp state    
        # TODO: what about opt_level, does it matter if it's different from before?
        # TODO: what if trained per-loss loss-scale and now using global or vice versa? Just reset it, i.e. ignore the amp state_dict?
        if self.opt.mixed_precision:
            if "amp" not in checkpoint:
                print("This checkpoint was not trained using mixed precision.")  # set as logger warning
            else:
                amp.load_state_dict(checkpoint['amp'])
        else:
            if "amp" in checkpoint:
                print("Loading a model trained with mixed precision without having initiliazed mixed precision")  # logger warning
        
        # load optimizers   
        # TODO: option not to load the optimizers. Useful if a training was completed or if scheduler brought optimizers' LR lower than wanted
        if self.is_train:
            self.optimizers['G'].load_state_dict(checkpoint['optimizer_G']) 
            self.optimizers['D'].load_state_dict(checkpoint['optimizer_D']) 


    def print_networks(self):
        """Print the total number of parameters in the network and network architecture"""
        print('---------- Networks initialized -------------')
        for name in self.networks.keys():
            num_params = 0
            for param in self.networks[name].parameters():
                num_params += param.numel()
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


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
