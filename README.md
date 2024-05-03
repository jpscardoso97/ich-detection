# The Effect of Noise on Hemorrhage Detection using CT Scan Imaging​

## Description
This project aims to understand if training a binary classification model (to detect intracranial hemorrhage) on images with a certain amount of added noise can improve the model's performance on generalized noisy images.

## Dataset
The dataset used in the project is a subset of the RSNA Intracranial Hemorrhage Detection. The original dataset consists of more than 25000 512x512 images in DICOM format. For this project 4000 samples were converted to a PNG dataset in Kaggle: [https://www.kaggle.com/datasets/jpscardoso/rsna-bme548-png](https://www.kaggle.com/datasets/jpscardoso/rsna-bme548-png).

## Physical Transformation Simulation
To simulate a real scenario different levels of noise were to the sinograms of the images using the `scikit-image` library. The noise levels used were 0.0, 0.25, 0.5, 0.75, 1.0 and 1.5. 
The sinograms were generated using the `radon` method, then noise was added using the `skimage.util.random_noise` function and finally the new simulated CT scan was obtained through the `iradon` method which uses the filtered back projection algorithm to compute the inverse Radon transform.

![Pipeline](/images/pipeline.png)

## Code Structure
The code is divided into 3 main parts:
1. **Data Generation**: 
   * The dataset converted to PNG from DICOM
   * Train test split structure with folder per classification class is created
   * Noisy data sets are generated using the `scikit-image` library  
The pre-processing steps and the generation of noisy images, are implemented in the Jupyter notebooks in the [notebooks/pre-processing](notebooks/pre-processing) folder.

2. **Model Training and Evaluation**: The model is trained using a specific noise level and tested on every test set (each noise level). Refer to: [notebooks/evaluation/train_and_evaluate.ipynb](notebooks/evaluation/train_and_evaluate.ipynb).

3. **Physical Transformation Simulation**: The physical transformation simulation methods are implemented in scripts that are then imported in the previously mentioned Jupyter notebooks:
    * [scripts/physical_transformation.py](scripts/physical_transformation.py): Contains the methods to generate the sinograms and the inverse Radon transform.
    * [scripts/noise_simulation.py](scripts/noise_simulation.py): Contains the methods to add noise to the sinograms.

4. **Convolutional Neural Network**: The implementation of the VGG16 used in the project is in [scripts/model.py](scripts/model.py).
