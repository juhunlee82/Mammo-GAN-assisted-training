# Mammo-GAN-assisted-training
## Core idea
The performance of lesion detection algorithm is determined by how well it can learn the key features that distinguish challenging or borderline cases. 
The performance could be increased if there is more challenging cases in the training set.
Our purpose was therefore to increase the challengning or difficult cases by coverting existing training data with Cycle-GAN based Lesion Simulator (LS) and Lesion Remover (LR).
The figure below illustrates our key idea, increasing difficult samples by using LS and LR to convert the portion of easy lesions and easy normal tissue (a and c in Figure 1.A) as well as the difficult cases (b in Figure 1.A) to be more challenging. By doing so, we modified the score distributions from A) to C), such that the algorithm could learn key features for classification better by accessing more difficult or challenging cases. 

In fact, we can control the degree of occultness by undertraining LS and LR. Note that typical training of deep networks involve a few hundreds of epochs for training from scratch, and 10 – 100 for fine-tuning the networks. Some intermediate epochs, for examples 25, 50 or 75 epoches, can provide less occult lesions that could be served as key lesion cases to improve the performance of algorithms. Same scenarios can apply to normal cases.

<p align="center">
<img src="https://github.com/user-attachments/assets/b274cc94-382e-4155-9a5b-8e5a41883a59" width="80%" \>
  <figcaption> Figure 1. Core-idea of our Mammo-GAN assisted training method</figcaption>
</p>

## What this repository offer
This repository offer the weights of Lesion Simulator (LS) and Lesion Remover(LR) at different epochs, including 25, 50, 75, 100 for others to test our LS and LR methods to convert the existing training data to be more challenging. Below image illustrates how LS and LR could covert easy normal to challening normal (lesion-like but normal), and easy lesion to challening lesion (normal-like but lesion). We found Epoch 50 and 75 work best for lesion detection in mammograms and chest X-ray images.
<p align="center">
<img src="https://github.com/user-attachments/assets/fd61016e-9c8b-4e0b-a468-e07e7286a500" width="50%" \>
  <figcaption> Figure 2. How LS and LR works at different epoch </figcaption>
</p>

## How to use Lesion Simulator and Lesion Remover
First download original Cycle-GAN repository available at:
[Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Download our weights available at: [GAN-weights](https://drive.google.com/file/d/18c4AKCZmC6w5VfT5_LRvox3XPbsW3q8y/view?usp=drive_link)

You are in fact downloading generator weights for the original Cycle-GAN that were fine-tuned on the development portion (80%) of 10,414 mammography patch dataset from 4,789 unique patients (2,416 women with recalled lesions and 2,312 healthy women). 

There are two folders, /lesion_cycle_ganA/ and /lesion_cycle_ganB/, where the first is Lesion remover and the second one is Lesion simulator.
Each folder has weight for generator of Cycle-GAN, at epoch 25, 50, 75, and 100.

Save the above folders under \checkpoints\, if you don't have this folder, create one under the original Cycle-GAN repository.

You can apply the Lesion Simulator and Remover using the test.py from the original Cycle-GAN repository:

Lesion Remover:
```python
python test.py --dataroot [your datafolder] --name lesion_cycle_ganA --model test --no_dropout --num_test 5000 --preprocess none  --results_dir [your result folder]
```
Lesion Simulator:
```python
python test.py --dataroot [your datafolder] --name lesion_cycle_ganB --model test --no_dropout --num_test 5000 --preprocess none  --results_dir [your result folder]
```
## Test Lesion simulator and Lesion remover assisted training for mammograms and chest X-ray images
Mammogram patch dataset
First train baseline model:
```matlab
/Test_on_Mammograms/Baseline_training.m
```
After having the Lesion simulator and Lesion Remover processed patches ready, 
```matlab
/Test_on_Mammograms/LS_LR_assisted_training.m
```
Note that this code goes over various Threshold values to find the target training samples to be converted by the lesion remover and lesion simulator, and different epoch levels for lesion simulator and lesion remover should be operated. Please prepare the converted samples for different LS - LR epochs accessible for the code.   

chest X-ray patch dataset
First train baseline model:
```matlab
/Test_on_ChestXray/Baseline_training.m
```
After having the Lesion simulator and Lesion Remover processed patches ready, 
```matlab
/Test_on_ChestXray/LS_LR_assisted_training.m
```
