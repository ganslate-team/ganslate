from loguru import logger

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def get_single_image_transform(conf, method=Image.BICUBIC):
    """
    Returns a callable transform composition that acts on a single image. Useful during unpaired training.
    """
    preprocess = conf[conf.mode].dataset.preprocess
    load_size = conf[conf.mode].dataset.load_size
    final_size = conf[conf.mode].dataset.final_size
    image_channels = conf[conf.mode].dataset.image_channels

    transform_list = []

    # Initial resizing transforms
    if 'resize' in preprocess:
        transform_list.append(transforms.Resize(load_size, method))

    elif 'scale_width' in preprocess:
        load_w = load_size[1]
        final_w = final_size[1]
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, load_w, final_w, method)))

    # elif 'scale_shortside' in preprocess:
    #     transform_list.append(
    #         transforms.Lambda(lambda img: __scale_shortside(img, load_size, final_size, method)))


    # Random transforms for data augmentation
    if 'random_zoom' in preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __random_zoom(img, final_size, method)))

    if 'random_crop' in preprocess:
        transform_list.append(transforms.RandomCrop(final_size))

    # if 'patch' in preprocess:
    #     transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], final_size)))

    # if 'random_trim' in preprocess:
    #     transform_list.append(transforms.Lambda(lambda img: __trim(img, final_size)))

    if 'random_flip' in preprocess:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]

    if image_channels == 1:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    elif image_channels == 3:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        raise ValueError("Transforms support `image_channels` set to 1 or 3.")

    return transforms.Compose(transform_list)


def get_paired_image_transform(conf, method=Image.BICUBIC):
    """
    Returns a callable transform composition that operates simultaneously and uniformly on an A-B image pair.
    """
    preprocess = conf[conf.mode].dataset.preprocess
    load_size = conf[conf.mode].dataset.load_size
    final_size = conf[conf.mode].dataset.final_size
    image_channels = conf[conf.mode].dataset.image_channels
    mode = conf.mode

    if image_channels == 1:
        normalize = transforms.Normalize((0.5,), (0.5,))
    elif image_channels == 3:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError("Transforms support `image_channels` set to 1 or 3.")

    # If mode is val/test and if random transforms are given, skip them
    if mode != 'train' and any(['random_' in tfm for tfm in preprocess]):
        preprocess = [tfm for tfm in preprocess if 'random_' not in tfm]
        raise Warning(f"Random transform(s) given in the preprocess list in the `{mode}` mode. \
            These tranmsforms will not be applied in this mode.") 
        
    def paired_image_transform(A_img, B_img):

        load_h, load_w = load_size
        final_h, final_w = final_size
        
        # Initial resizing transforms
        if 'resize' in preprocess:
            A_img = TF.resize(A_img, load_size, method)
            B_img = TF.resize(B_img, load_size, method)

        elif 'scale_width' in preprocess:
            A_img = __scale_width(A_img, load_w, final_w, method)
            B_img = __scale_width(B_img, load_w, final_w, method)

        # Random transforms for data augmentation
        if 'random_zoom' in preprocess:
            zoom_level = tuple(np.random.uniform(0.8, 1.0, size=[2]))        
            A_img = __random_zoom(A_img, final_size, method, zoom_level)
            B_img = __random_zoom(B_img, final_size, method, zoom_level)

        if 'random_crop' in preprocess:
            top, left = np.random.randint(load_h - final_h) , np.random.randint(load_w - final_w)           
            A_img = TF.crop(A_img, top, left, final_h, final_w)
            B_img = TF.crop(B_img, top, left, final_h, final_w)
        
        if 'random_flip' in preprocess:
            if np.random.choice(['flip', 'no_flip']) == 'flip':
                A_img = TF.hflip(A_img)
                B_img = TF.hflip(B_img)

        A_img = TF.to_tensor(A_img)
        B_img = TF.to_tensor(B_img)

        A_img = normalize(A_img)
        B_img = normalize(B_img)

        return A_img, B_img

    return paired_image_transform

        
# ---

def __make_power_2(img, base, method=Image.BICUBIC):
    img_w, img_h = img.size
    new_h = int(round(img_h / base) * base)
    new_w = int(round(img_w / base) * base)
    if new_h == img_h and new_w == img_w:
        return img

    return img.resize((new_w, new_h), method)


def __random_zoom(img, final_size, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    img_w, img_h = img.size
    final_h, final_w = final_size
    zoom_w = max(final_w, img_w * zoom_level[0])
    zoom_h = max(final_h, img_h * zoom_level[1])
    img = img.resize((int(round(zoom_w)), int(round(zoom_h))), method)
    return img


# def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
#     ow, oh = img.size
#     shortside = min(ow, oh)
#     if shortside >= target_width:
#         return img
#     else:
#         scale = target_width / shortside
#         return img.resize((round(ow * scale), round(oh * scale)), method)


# def __trim(img, trim_width):
#     ow, oh = img.size
#     if ow > trim_width:
#         xstart = np.random.randint(ow - trim_width)
#         xend = xstart + trim_width
#     else:
#         xstart = 0
#         xend = ow
#     if oh > trim_width:
#         ystart = np.random.randint(oh - trim_width)
#         yend = ystart + trim_width
#     else:
#         ystart = 0
#         yend = oh
#     return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, load_w, final_w, method=Image.BICUBIC):
    img_w, img_h = img.size
    if img_w == load_w and img_w >= final_w:
        return img
    scaled_w = load_w
    scaled_h = int(max(load_w * img_h / img_w, final_w))
    return img.resize((scaled_w, scaled_h), method)


def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        logger.info("The image size needs to be a multiple of 4. "
                    "The loaded image size was (%d, %d), so it was adjusted to "
                    "(%d, %d). This adjustment will be done to all images "
                    "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
