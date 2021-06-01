import torch
from torch import fft

def get_freq_transform(image):
    """
    Returns tanh(magnitude spectrum) of the FFT transform of an image
    """
    image = (image + 1)/2
    # Half precision needs to be converted to single precision
    # due to lack of support in pytorch for half-fft
    # https://github.com/pytorch/pytorch/issues/42175#issuecomment-677665333
    if isinstance(image, torch.HalfTensor) or isinstance(image, torch.cuda.HalfTensor):
        image = image.float()

    f_image = fft.fftn(image, norm='ortho')
    f_image = fft.fftshift(f_image)
    f_image = torch.abs(f_image)
    f_image = torch.tanh(f_image)
    return f_image
        
