# Successive Refinement in Large-Scale Computation
This projects explores solutions for massive computation with layered resolution. as described in the paper "Successive Refinement in Large-Scale Computation: Expediting Model Inference Applications".

# Project Structure

## Succcessive Refinement for CIFAR10 classification using a trained VGG16 model architecture
The folder "Successice_Refinement_CIFAR10_VGG16" containes the necessary codes.

Download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset Python version. Unzip the file and place the "cifar-10-batches-py" folder somewhere accessible.
Install miniconda if you haven't already.
The dependencies will be later uploaded. Install the dependencies by running "conda env create -f environment.yml" inside the root of this project.
It will create a new environment and install all dependencies.

### Description of the files
- train_cifar10_VGG16.py: A python code for training a VGG16 model (with full resolution) on CIFAR 10. The trained model will be saved in a folder called checkpoints.
- train_cifar10_VGG16.sh: A script for calling the training of VGG16 model on CIFAR 10, given the desired parameters
- test_cifar10_VGG16.sh: A script for calling the testing of a trained VGG16 model on CIFAR 10, given the desired parameters
- models.py: Contained the necessary model architectures
- main.py: The main python code for performing an inference task on a trained VGG16 model with successive refinement.

For running these simulations, one first needs to train a VGG16 model using 
./train_cifar10_VGG16.sh
and then perform the inference with successice refinement using
python main.py'

## Succcessive Refinement for MNIST classification using a trained Dense Neural Network
The folder "Successice_Refinement_MNIST_FCNN" containes the necessary codes.
The dataset will be automatically downloaded the first time running the code.
All codes are in a signle python file that can be run as 
python paper_based_version.py
This version of the code will be integrated with the above folder in future.

## Succcessive Refinement for an Stream of matrix-vector multiplications over distributed workers
The folder "Successice_Refinement_StreamMatrixVectorMultiplication" containes the necessary codes.
All codes under this part of the project are written in Matlab
### Description of the files
- optimal_load_split.m: A code for load partitioning (scheduling) among the workers
- 
