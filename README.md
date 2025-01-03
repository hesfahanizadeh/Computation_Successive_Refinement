# Successive Refinement in Large-Scale Computation
This project explores solutions for massive computation with layered resolution, as described in the paper "Successive Refinement in Large-Scale Computation: Expediting Model Inference Applications."

# Project Structure

## Successive Refinement for CIFAR-10 Classification Using a Trained VGG16 Model
The folder "Successice_Refinement_CIFAR10_VGG16" containes the necessary codes.

Download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset Python version. Unzip the file and place the "cifar-10-batches-py" folder somewhere accessible.
Install miniconda if you haven't already.
[The dependencies will be later uploaded.] 
Install the dependencies by running "conda env create -f environment.yml" inside the root of this project.
It will create a new environment and install all dependencies.

### Description of the files
- train_cifar10_VGG16.py: Python script to train a VGG16 model (full resolution) on CIFAR-10. The trained model is saved in the checkpoints folder.
- train_cifar10_VGG16.sh: Script to train the VGG16 model with specified parameters.
- test_cifar10_VGG16.sh: Script to test a trained VGG16 model with specified parameters.
- models.py: Contains the model architectures.
- main.py: Main Python script to perform inference on a trained VGG16 model with successive refinement.

To Run:
1. Train the VGG16 model:
./train_cifar10_VGG16.sh
2. Perform inference with successive refinement:
python main.py

## Successive Refinement for MNIST Classification Using a Dense Neural Network
The folder "Successice_Refinement_MNIST_FCNN" containes the necessary codes.
The MNIST dataset is automatically downloaded when the code runs for the first time.
All code is in a single Python file: paper_based_version.py.
To Run:
python paper_based_version.py
Note: This version will be integrated with the CIFAR-10 folder in future updates.

## Successive Refinement for a Stream of Matrix-Vector Multiplications over Distributed Workers
The folder "Successice_Refinement_StreamMatrixVectorMultiplication" contains the necessary Matlab codes.
### Description of the files
- optimal_load_split.m: Function for load partitioning (scheduling) among workers.
- Analysis_Omega.m: Evaluates delay vs. computational redundancy for layered-resolution computations.
- Analysis_Deadline.m: Analyzes success rate vs. deadline for layered-resolution computations.
- Analysis_Deadline_Omega.m: Evaluates success rate vs. computational redundancy at various resolution layers.

