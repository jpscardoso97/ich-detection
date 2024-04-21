import numpy as np

def calculate_snr(original_image, noisy_image):
    # Calculate the mean signal value of the original image
    mean_signal = np.mean(original_image)
    
    # Calculate the noise image by subtracting the original image from the noisy image
    noise_image = noisy_image - original_image
    
    # Calculate the standard deviation of the noise image
    std_noise = np.std(noise_image)
    
    # Calculate the SNR
    snr = mean_signal / std_noise if std_noise != 0 else float('inf')
    
    return snr