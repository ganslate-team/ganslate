import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
import torch.nn.functional as F
from pytorch_msssim.ssim import SSIM, MS_SSIM
from apex import amp


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
        # number of losses on which backward is performed, G, D_A, D_B (3).
        self.num_losses = 3
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'inv_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'inv_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.is_train and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.is_train:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        # -------------------------- Network initialization --------------------------
        # load/define networks3d
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks3d.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, opt.use_naive,
                                        opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = lambda x: self.netG(x)
        self.netG_B = lambda x: self.netG(x, inverse=True)

        if self.is_train:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks3d.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                              opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks3d.define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                              opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
        # ----------------------------------------------------------------------------

        # -------------------------- Losses and optimizers ---------------------------
        if self.is_train:
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN   = networks3d.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionSSIM  = SSIM(data_range=(-1,1), channel=opt.patch_size[0])
            self.criterionIdt   = torch.nn.L1Loss()
            self.criterionInv   = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # ----------------------------------------------------------------------------

        self.setup() # schedulers, mixed precision, checkpoint loading and network parallelization

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict) -- include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        # Forward cycle (A to B)
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A  = self.netG_B(self.fake_B)

        # Backward cycle (B to A)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B  = self.netG_A(self.fake_A)


    def backward_D_basic(self, netD, real, fake, loss_id=0):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if self.opt.grad_reg > 0.0:
            self.backward(loss_D, self.optimizer_D, retain_graph=True, loss_id=loss_id)
            Lgrad = torch.cat([x.grad.view(-1) for x in netD.parameters()])
            loss_D = loss_D - self.opt.grad_reg * (0.5 * torch.norm(Lgrad) / len(Lgrad))
        else:
            self.backward(loss_D, self.optimizer_D, retain_graph=True, loss_id=loss_id)
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        if self.opt.per_loss_scale: 
            loss_id = 1  # D_A is 1, D_B is 2, G is 0
        else:
            loss_id = 0

        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, loss_id)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        if self.opt.per_loss_scale:
            loss_id = 2  # D_A is 1, D_B is 2, G is 0
        else:
            loss_id = 0
            
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, loss_id)
        

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B using all specified losses"""
        loss_id = 0  # D_A is 1, D_B is 2, G is 0
        lambda_idt = self.opt.lambda_identity
        lambda_inv = self.opt.lambda_inverse
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        proportion_SSIM = self.opt.proportion_ssim

        # ---------------------- Identity Loss -------------------------
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt    
        # ---------------------------------------------------------------        
        
        # ----------------------- Inverse Loss -------------------------
        # TODO: Find a better name
        self.loss_inv_A = 0
        self.loss_inv_B = 0  
        if lambda_inv > 0:
            self.loss_inv_A = self.criterionInv(self.real_A, self.fake_B) * lambda_A * lambda_inv
            self.loss_inv_B = self.criterionInv(self.real_B, self.fake_A) * lambda_B * lambda_inv
        # ---------------------------------------------------------------

        # ------------------------- GAN Loss ----------------------------
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) # Forward GAN loss D_A(G_A(A))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) # Backward GAN loss D_B(G_B(B))
        # ---------------------------------------------------------------

        # ---------------- Cycle-consistency Loss -----------------------

        # L1 (standard for cycle-consistency) 
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) # Forward cycle loss  || G_B(G_A(A)) - A||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) # Backward cycle loss || G_A(G_B(B)) - B||
        
        # Cycle-consistency using a weighted combination of SSIM and L1 
        if proportion_SSIM > 0:
            alpha, beta = proportion_SSIM, 1-proportion_SSIM  # weights for addition of SSIM and L1 losses
            # (1-SSIM) because the closer the images are, the higher value will SSIM give (max 1, min -1)
            ssim_loss_A       = 1 - self.criterionSSIM(self.rec_A, self.real_A) 
            ssim_loss_B       = 1 - self.criterionSSIM(self.rec_B, self.real_B)
            # weighted sum of SSIM and L1 losses for both forward and backward cycle losses
            self.loss_cycle_A = alpha * ssim_loss_A + beta * self.loss_cycle_A  
            self.loss_cycle_B = alpha * ssim_loss_B + beta * self.loss_cycle_B 
            
        self.loss_cycle_A = self.loss_cycle_A * lambda_A 
        self.loss_cycle_B = self.loss_cycle_B * lambda_B
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
                      + self.loss_idt_A + self.loss_idt_B + self.loss_inv_A + self.loss_inv_B
        
        self.backward(self.loss_G, self.optimizer_G, loss_id)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()  # compute fake images and reconstruction images.
        
        # ------------------------ G_A and G_B -------------------
        self.set_requires_grad([self.netD_A, self.netD_B], False)   # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # --------------------------------------------------------

        # ------------------------ D_A and D_B -------------------
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   #set D_A and D_B's gradients to zero
        self.backward_D_A()     # calculate gradients for D_A
        self.backward_D_B()     # calculate graidents for D_B
        self.optimizer_D.step() # update D_A and D_B's weights
        # --------------------------------------------------------
