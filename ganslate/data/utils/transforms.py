from loguru import logger

import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def get_transform(conf, method=Image.BICUBIC):
    preprocess = conf[conf.mode].dataset.preprocess
    load_size = conf[conf.mode].dataset.load_size
    crop_size = conf[conf.mode].dataset.crop_size
    flip = conf[conf.mode].dataset.flip
    image_channels = conf[conf.mode].dataset.image_channels

    transform_list = []

    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))

    elif 'scale_width' in preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))

    elif 'scale_shortside' in preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_shortside(img, load_size, crop_size, method)))

    if 'zoom' in preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __random_zoom(img, load_size, crop_size, method)))

    if 'crop' in preprocess:
        transform_list.append(transforms.RandomCrop(crop_size))

    # if 'patch' in preprocess:
    #     transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], crop_size)))

    if 'trim' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, crop_size)))

    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]

    if image_channels == 1:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    elif image_channels == 3:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        raise ValueError("Transforms support `image_channels` set to 1 or 3.")

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img


def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)


def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method)


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
