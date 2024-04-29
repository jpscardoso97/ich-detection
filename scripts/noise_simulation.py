import numpy as np
from skimage.util import random_noise
from enum import Enum

class NoiseType(Enum):
    GAUSSIAN = 1
    POISSON = 2
    SALT_AND_PEPPER = 3
    SPECKLE = 4

"""
 This class defines the noise configuration to be applied to the CT scan dataset.
"""
class NoiseConfiguration:
    def __init__(self, noise_types, noise_levels):
        """
        Initialize the noise configuration.

        Args:
            noise_types (list): A list of noise types to apply. Supported types are NoiseType enums.
            noise_levels (list): A list of noise levels to apply, corresponding to the noise types.
        """
        self.noise_types = noise_types
        self.noise_levels = noise_levels

"""
This class simulates different types of noise on CT scan images.
"""
class NoiseSimulator:
    @staticmethod
    def apply_gaussian_noise(image, sigma):
        """
        Apply Gaussian noise to the input image.

        Gaussian noise is a common type of noise that can be observed in medical imaging due to various factors, such as electronic noise in the imaging device or thermal noise. This type of noise is characterized by a normal distribution of pixel values around the original pixel value.

        Args:
            image (numpy.ndarray): The input image to be transformed.
            noise_level (float): The standard deviation of the Gaussian noise to be applied.

        Returns:
            numpy.ndarray: The image with Gaussian noise applied.
        """
        noisy_image = random_noise(image, mode="gaussian", var=sigma**2, clip=True)

        return noisy_image

    @staticmethod
    def apply_poisson_noise(image):
        """
        Apply Poisson noise to the input image.

        Poisson noise is another type of noise that is common in medical imaging, especially in low-light conditions or when the signal-to-noise ratio is low.
        This type of noise is characterized by a Poisson distribution of pixel values, where the variance of the noise is proportional to the original pixel value.

        Args:
            image (numpy.ndarray): The input image to be transformed.

        Returns:
            numpy.ndarray: The image with Poisson noise applied.
        """
        
        return random_noise(image, mode="poisson", clip=True)



    @staticmethod
    def apply_salt_and_pepper_noise(image, noise_pct):
        """
        Apply salt-and-pepper noise to the input image.

        Salt-and-pepper noise is a type of noise that can be observed in medical imaging due to sensor errors or bit errors during data transmission. 
        This type of noise is characterized by randomly occurring white and black pixels, which can be interpreted as "salt" (white) and "pepper" (black) pixels.

        Args:
            image (numpy.ndarray): The input image to be transformed.
            noise_pct (float): The proportion of pixels to apply Poisson noise to (0-1).

        Returns:
            numpy.ndarray: The image with salt-and-pepper noise applied.
        """

        return random_noise(image, mode="s&p", amount=noise_pct)


    @staticmethod
    def apply_speckle_noise(image, noise_level):
        """
        Apply speckle noise to the input image.

        Speckle noise is a type of multiplicative noise that is common in ultrasound imaging. This type of noise is characterized by a granular pattern that can be observed in the image due to interference between the transmitted and reflected signals.

        Args:
            image (numpy.ndarray): The input image to be transformed.
            noise_level (float): The variance of the speckle noise to be applied.

        Returns:
            numpy.ndarray: The image with speckle noise applied.
        """

        return random_noise(image, mode="speckle", var=noise_level**2)
    

    @staticmethod
    def apply_noise(image: np.ndarray, noise_config: NoiseConfiguration):
        """
        Apply the specified noise configurations to the input image.

        Args:
            image (numpy.ndarray): The input image to be transformed.
            noise_config (NoiseConfiguration): The noise configuration object that defines the noise types and levels to be applied.

        Returns:
            numpy.ndarray: The image with the specified noise applied.
        """
        noisy_image = image.copy()
        for noise_type, noise_level in zip(noise_config.noise_types, noise_config.noise_levels):
            if noise_type == NoiseType.GAUSSIAN:
                noisy_image = NoiseSimulator.apply_gaussian_noise(noisy_image, noise_level)
            elif noise_type == NoiseType.POISSON:
                noisy_image = NoiseSimulator.apply_poisson_noise(noisy_image)
            elif noise_type == NoiseType.SALT_AND_PEPPER:
                noisy_image = NoiseSimulator.apply_salt_and_pepper_noise(noisy_image, noise_level)
            elif noise_type == NoiseType.SPECKLE:
                noisy_image = NoiseSimulator.apply_speckle_noise(noisy_image, noise_level)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            
        return noisy_image