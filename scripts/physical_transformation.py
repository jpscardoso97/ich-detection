import numpy as np

from scripts.noise_simulation import NoiseConfiguration, NoiseSimulator, NoiseType
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import radon, iradon

def get_noisy_image(image, noise_level):
    subsample = 2
    theta = np.linspace(0., 180., max(image.shape)//subsample, endpoint=False)
    sinogram = radon(image, theta=theta)

    if noise_level == 0:
        return iradon(sinogram, theta=theta, filter_name='ramp')

    simulation_config = NoiseConfiguration([NoiseType.GAUSSIAN], [noise_level])
    noisy_sinogram = NoiseSimulator.apply_noise(sinogram, simulation_config)
    return iradon(noisy_sinogram, theta=theta, filter_name='ramp')


def get_direct_noisy_image(image, noise_level):
    simulation_config = NoiseConfiguration([NoiseType.GAUSSIAN], [noise_level])
    return NoiseSimulator.apply_noise(image, simulation_config)
    