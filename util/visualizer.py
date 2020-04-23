import os
import ntpath
import time
import numpy as np
import torch
from . import util
from . import html
from PIL import Image

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, scale_range, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        # Stack first dimension so greyscale (e.g. HU) becomes greyscale rgb
        if len(image_numpy.shape) == 3:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if len(image_numpy.shape) == 4:
            image_numpy = np.tile(image_numpy, (3, 1, 1, 1))

    if len(image_numpy.shape) == 4:
        # slice 3d volume in the middle to visualize 2d slice
        image_numpy = image_numpy[:3, image_numpy.shape[1] // 2, :, :]
        # image_numpy = image_numpy[:3, :, image_numpy.shape[2] // 2, :]

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - scale_range[0]) / (scale_range[1] - scale_range[0]) * 255.0

    # Limit image range (can happen without output tanh)
    image_numpy[image_numpy < 0] = 0
    image_numpy[image_numpy > 255] = 255

    return image_numpy.astype(imtype)

# save image to the disk
def save_images(webpage, visuals, image_path, scale_range, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data, scale_range=scale_range)
        parsname = name.replace('/', '_')
        image_name = '%s_%s.png' % (parsname, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = Image.fromarray(im).resize( (h, int(w * aspect_ratio)) )
        if aspect_ratio < 1.0:
            im = Image.fromarray(im).resize( (int(h / aspect_ratio), w) )
        im.save(save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, opt):
        self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.scale_range = (-1, 1)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.use_html and not self.saved:  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = tensor2im(image, self.scale_range)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                im = Image.fromarray(image_numpy)
                im.save(img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = tensor2im(image, self.scale_range)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
