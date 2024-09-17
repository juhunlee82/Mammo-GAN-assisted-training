# Mammo-GAN-assisted-training
## Core idea
The performance of lesion detection algorithm is determined by how well it can learn the key features that distinguish challenging or borderline cases. 
The performance could be increased if there is more challenging cases in the training set.
Our purpose was therefore to increase the challengning or difficult cases by coverting easy cases with Cycle-GAN based Lesion Simulator (LS) and Lesion Remover (LR).
The figure below illustrates our key idea, increasing difficult samples by using LS and LR to convert easy lesions and easy normal tissue (a and c in Figure 1.A) to difficult cases (i.e., moving them in the area b in Figure 1.C). By doing so, we modified the score distributions from A) to C), such that the algorithm could learn key features for classification better by accessing more difficult or challenging cases. 

In fact, we can control the degree of occultness by undertraining LS and LR. Note that typical training of deep networks involve a few hundreds of epochs for training from scratch, and 10 – 100 for fine-tuning the networks. Some intermediate epochs, for examples 25, 50 or 75 epoches, can provide less occult lesions that could be served as key lesion cases to improve the performance of algorithms. Same scenarios can apply to normal cases.

<p align="center">
<img src="https://github.com/user-attachments/assets/e1e90450-b173-459d-b0dd-3f09c7c2e027" width="60%" \>
</p>

## What this repository offer
This repository offer the weights of Lesion Simulator (LS) and Lesion Remover(LR) at different epochs, including 25, 50, 75, 100 for others to test our LS and LR methods to convert the easy cases to challenging or borderline cases. Below image illustrates how LS and LR could covert easy normal to challening normal (lesion-like but normal), and easy lesion to challening lesion (normal-like but lesion). We found Epoch 50 and 75 work best for lesion detection in mammograms and chest X-ray images.
<p align="center">
<img src="https://github.com/user-attachments/assets/fd61016e-9c8b-4e0b-a468-e07e7286a500" width="50%" \>
</p>

## How to use Lesion Simulator and Lesion Remover
First download original Cycle-GAN repository available at:
[Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Download our weights. You are in fact downloading generator weights for the original Cycle-GAN that were fine-tuned on the development portion (80%) of 10,414 mammography patch dataset from 4,789 unique patients (2,416 women with recalled lesions and 2,312 healthy women). 

We adopted the original [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and fine-tuned it to develop our Lesion Simulator (LS) and Lesion Remover(LR)

What this repository offer are the weights of LS and LR at different epoch

Using the lesion locations marked by MQSA radiologists, we segmented the patches to a size of 400 by 400 pixels (2.8 cm by 2.8 cm in size), including the recalled lesions for the cases. For normal controls, we segmented the same 400 by 400 pixel patch from the centroid of the breast area.

<p align="center">
<img src="https://github.com/user-attachments/assets/541b77fe-8319-4c47-8c29-9aae587a59c4" width="50%" \>
</p>

Our LS and LR are the generators of Cycle-GAN, which can insert lesion in the normal patch (LS) and remove lesion from the lesion patch (LR). 
We optimized the Cycle-GAN using an Adam optimizer with a learning rate of 0.0002, and momentum parameters of β1=0.5, β2=0.999. In addition, we set the maximum epoch as 100 and saved the model at every 5 eporch, and the weights for L1 regularization, λ1 and λ2, as 10 and 0.5, and a minibatch size of 4. We used a random left-right vertical flip as data augmentation.
Below figure shows how LS and LR work for the course of training. We prepared the LS and LR weights for epoch 5, 10, 15, ..., 100 to this repository.





<p align="center">
<img src="https://github.com/user-attachments/assets/fd61016e-9c8b-4e0b-a468-e07e7286a500" width="50%" \>
</p>

## Creating challenging cases for improving the detection performance 
Our hypothesis is that LS and LR modified samples could be served as difficult or challenging samples for the lesion detection algorithm, such that they can ultimately improve the lesion detection performance once the algorithm trained on such modified cases. In addition, we assumed that LS and LR in intermediate training steps can provide better challenging cases than those from the final training steps.  
Once the detection algorithm is optimized on the training set, the lesion score distribution by the algorithm can be ranged from 0 (normal) to 1 (recalled lesion) as shown in Figure 1.A. Let Th be the positive threshold value to determine the easy cases; the normal patches with the score lower than +Th (distribution in blue, or a in Figure 1.A) and lesion patches with the score higher than 1 – Th (distribution in orange, or c in Figure 1.A) are easy samples for the detection algorithm. Any cases within +Th and 1 – Th (distribution in green or b in Figure 1.A) are difficult or challenging cases for the algorithm. Easy normal cases are then passed through the LS to increase the difficulty of those cases. Likewise, we increased the difficulty level of easy lesions by feeding them into LR. We then replaced such transformed cases back with the original easy cases (Figure 1.B). As a result, the number of challenging or difficult cases can be increased for retraining the given algorithm (Figure 1.C).  

To use LS and LR for improving lesion detection algorithms, one needs to decide how many easy samples should be changed and which epoch of LS and LR should be used. To do so, we considered the following Th levels of [0.05, 0.1, 0.25, 0.5, 0.75, 1] and for Cycle-GAN epochs, we evaluated the cases of 25, 50, 75, and 100 epochs.  
