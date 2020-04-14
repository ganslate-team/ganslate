import os
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict
from . import networks
from util.util import remove_module_from_ordered_dict
from apex.parallel import DistributedDataParallel
from apex import amp


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
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
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        # number of losses on which backward is performed, G_A, G_B, D_A, D_B (4);
        # in case of partially invertible, it is G, D_A, D_B (3).
        self.num_losses = 4  

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser
    
    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
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
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        if self.opt.mixed_precision:
            self.convert_to_mixed_precision(self.opt.opt_level, self.opt.per_loss_scale)

        if not self.is_train or self.opt.continue_train:
            self.load_networks(self.opt.epoch)

        if len(self.gpu_ids) > 1:
            self.parallelize_networks()
        
        torch.cuda.empty_cache()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

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
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if self.opt.distributed:
                    net = DistributedDataParallel(net)
                else:
                    net = torch.nn.DataParallel(net, self.gpu_ids)

    def convert_to_mixed_precision(self, opt_level='O1', per_loss_scale=False):
        """Initializes Nvidia Apex Mixed Precision
        Parameters:
            opt_level (str) -- specifies the AMP's optimization level. Accepted values are
                               "O0", "O1", "O2", and "O3". Check Apex documentation.
            num_losses -- Option to tell Amp in advance how many losses/backward passes you plan to use. 
                          When used in conjunction with the `loss_id` argument to amp.scale_loss, enables Amp to use 
                          a different loss scale per loss/backward pass, which can improve stability.
                          If `num_losses=1`, Amp will still support multiple losses/backward passes, 
                          but use a single global loss scale for all of them.
        """
        # fetch the networks
        networks = []
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                networks.append(net)

        if per_loss_scale:
            num_losses = self.num_losses
        else:
            num_losses = 1

        # initialize mixed precision on networks and, if training, on optimizers
        if self.is_train:
            networks, [self.optimizer_G, self.optimizer_D] = amp.initialize(
                                                                networks, 
                                                                [self.optimizer_G, self.optimizer_D], 
                                                                opt_level=opt_level, 
                                                                num_losses=num_losses
                                                             )
            self.optimizers = [self.optimizer_G, self.optimizer_D]
        else:
            networks = amp.initialize(networks, opt_level=opt_level)

        # assigns back the returned mixed-precision networks # TODO: say why
        for i, name in enumerate(self.model_names):
            if isinstance(name, str):
                setattr(self, 'net' + name, networks[i])

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        """ Return current learning rates of both generator and discriminator"""
        lr_G = self.optimizers[0].param_groups[0]['lr']
        lr_D = self.optimizers[1].param_groups[0]['lr']
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
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net, DistributedDataParallel):
                    checkpoint[name] = net.module.state_dict()  # e.g. checkpoint["D_A"]
                else:
                    checkpoint[name] = net.state_dict()

        # add optimizers to checkpoint
        checkpoint["optimizer_G"] = self.optimizer_G.state_dict()
        checkpoint["optimizer_D"] = self.optimizer_D.state_dict()

        # save apex mixed precision
        if self.opt.mixed_precision:
            checkpoint["amp"] = amp.state_dict()

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
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.load_state_dict(checkpoint[name])

        # load amp state    
        # TODO: what about opt_level, does it matter if it's different from before?
        # TODO: what if trained per-loss loss-scale and now using global or vice versa? Just reset it, i.e. ignore the amp state_dict?
        if self.opt.mixed_precision:
            if "amp" not in checkpoint:
                print("This checkpoint was not trained using mixed precision.")  # set as logger warning
            else:
                amp.load_state_dict(checkpoint["amp"])
        else:
            if "amp" in checkpoint:
                print("Loading a model trained with mixed precision without having initiliazed mixed precision")  # logger warning
        
        # load optimizers
        if self.is_train:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"]) 
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"]) 


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
