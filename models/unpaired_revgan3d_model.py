import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
import torch.nn.functional as F
from pytorch_msssim.ssim import SSIM, MS_SSIM
from apex import amp

from models.losses.losses import GeneratorLosses
from models.losses.GAN_loss import GANLoss

class UnpairedRevGAN3dModel(BaseModel):
    ''' Unpaired 3D-RevGAN model '''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_inverse', type=float, default=0.0, help='use inverse mapping. Setting lambda_inverse other than 0 has an effect of scaling the weight of the inverse mapping loss. For example, if the weight of the inverse loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_inverse = 0.1')
            parser.add_argument('--proportion_ssim', type=float, default=0.0, help='TODO')
        return parser

    def __init__(self, opt):
        super(UnpairedRevGAN3dModel, self).__init__(opt)
        
        # Inputs and Outputs of the model
        visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_A', 'real_B', 'fake_A', 'rec_B', 'idt_B']
        self.visuals = {name: None for name in visual_names}

        # Losses used by the model
        loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'inv_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'inv_B']
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['G', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Generator and Discriminators
        network_names = ['G', 'D_A', 'D_B'] if self.is_train else ['G'] # during test time, only G
        self.networks = {name: None for name in network_names}

        # Initialize Generators and Discriminators
        self.init_networks(opt)

        if self.is_train:
            # Intialize loss functions (criterions) and optimizers
            self.init_criterions(opt)
            self.init_optimizers(opt)

            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

        self.setup() # schedulers, mixed precision, checkpoint loading and network parallelization


    def init_networks(self, opt):
        for name in self.networks.keys():
            if name.startswith('G'):
                # TODO: make define_G and _D nicer
                # TODO: move it to base_model 
                self.networks[name] = networks3d.define_G(opt.input_nc, opt.output_nc,
                                                opt.ngf, opt.which_model_netG, opt.norm, opt.use_naive,
                                                opt.init_type, opt.init_gain, self.gpu_ids)
            elif name.startswith('D'):
                use_sigmoid = opt.no_lsgan
                self.networks[name] = networks3d.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                              opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                raise ValueError('Network\'s name has to begin with either "G" if it is a generator, \
                                  or "D" if it is a discriminator.')


    def init_criterions(self, opt):
        # Standard GAN loss
        self.criterion_GAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
        # Generator-related losses -- Cycle-consistency, Identity and Inverse loss
        self.criterions_G = GeneratorLosses(opt)


    def init_optimizers(self, opt):
        params_D = itertools.chain(self.networks['D_A'].parameters(), 
                                    self.networks['D_B'].parameters())         

        self.optimizers['D'] = torch.optim.Adam(params_D, lr=opt.lr_D, 
                                                betas=(opt.beta1, 0.999))

        self.optimizers['G'] = torch.optim.Adam(self.networks['G'].parameters(),
                                            lr=opt.lr_G, betas=(opt.beta1, 0.999))                             


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict) -- include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB' # TODO: more pythonic name
        self.visuals['real_A'] = input['A' if AtoB else 'B'].to(self.device)
        self.visuals['real_B'] = input['B' if AtoB else 'A'].to(self.device)

    
    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Forward cycle G_A (A to B)
        fake_B = self.networks['G'](real_A) 
        rec_A  = self.networks['G'](fake_B)

        # Backward cycle G_B (B to A)
        fake_A = self.networks['G'](real_B, inverse=True) # G forward is G_A, G inverse forward is G_B
        rec_B  = self.networks['G'](fake_A, inverse=True)

        self.visuals.update({'fake_B': fake_B, 'rec_A': rec_A, 
                             'fake_A': fake_A, 'rec_B': rec_B})

    
    def backward_D(self, discriminator, loss_id=0):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        if discriminator == 'D_A':
            real = self.visuals['real_B']
            fake = self.visuals['fake_B']
            fake = self.fake_A_pool.query(fake)
            
        elif discriminator == 'D_B':
            real = self.visuals['real_A']
            fake = self.visuals['fake_A']
            fake = self.fake_B_pool.query(fake)

        pred_real = self.networks[discriminator](real)
        pred_fake = self.networks[discriminator](fake.detach())

        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        loss_D_fake = self.criterion_GAN(pred_fake, target_is_real=False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5  # combined loss

        # backprop
        self.backward(loss_D, self.optimizers['D'], retain_graph=True, loss_id=loss_id)
        return loss_D


    def backward_G(self, loss_id=0):
        """Calculate the loss for generators G_A and G_B using all specified losses"""        

        real_A = self.visuals['real_A']        
        real_B = self.visuals['real_B']       
        fake_A = self.visuals['fake_A']  # G_B(B)
        fake_B = self.visuals['fake_B']  # G_A(A)

        # ------------------------- GAN Loss ----------------------------
        pred_A = self.networks['D_A'](fake_B)  # D_A(G_A(A))
        pred_B = self.networks['D_B'](fake_A)  # D_B(G_B(B))
        self.losses['G_A'] = self.criterion_GAN(pred_A, target_is_real=True) # Forward GAN loss D_A(G_A(A))
        self.losses['G_B'] = self.criterion_GAN(pred_B, target_is_real=True) # Backward GAN loss D_B(G_B(B))
        # ---------------------------------------------------------------

        # ------------- G Losses (Cycle, Identity, Inverse) -------------
        if self.criterions_G.use_identity():
            self.visuals['idt_A'] = self.networks['G'](real_B)
            self.visuals['idt_B'] = self.networks['G'](real_A, inverse=True)
        losses_G = self.criterions_G(self.visuals)
        self.losses.update(losses_G)
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = sum(losses_G.values()) + self.losses['G_A'] + self.losses['G_B']
        self.backward(combined_loss_G, self.optimizers['G'], loss_id)


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights. 
        Called in every training iteration.
        """
        self.forward()  # compute fake images and reconstruction images.

        discriminators = [ self.networks['D_A'], self.networks['D_B'] ]
        # ------------------------ G (A and B) ----------------------------------------------------
        self.set_requires_grad(discriminators, False)   # Ds require no gradients when optimizing Gs
        self.optimizers['G'].zero_grad()                # set G's gradients to zero
        self.backward_G(loss_id=0)                      # calculate gradients for G
        self.optimizers['G'].step()                     # update G's weights
        # ------------------------ D_A and D_B ----------------------------------------------------
        self.set_requires_grad(discriminators, True)
        self.optimizers['D'].zero_grad()                #set D_A and D_B's gradients to zero
        self.backward_D('D_A', loss_id=1) # calculate gradients for D_A
        self.backward_D('D_B', loss_id=2) # calculate graidents for D_B
        self.optimizers['D'].step()                     # update D_A and D_B's weights
        # -----------------------------------------------------------------------------------------
