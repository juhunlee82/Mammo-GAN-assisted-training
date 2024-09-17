# Mammo-GAN-assisted-training
The performance of lesion detection algorithm is determined by how well it can learn the key features that distinguish challenging or borderline cases. 
The performance could be increased if there is more challenging cases in the training set.
Our purpose was therefore to increase the challengning or difficult cases by coverting easy cases with Cycle-GAN based Lesion Simulator (LS) and Lesion Remover (LR).
The figure below illustrates our key idea, increasing difficult samples by using LS and LR to convert easy lesions and easy normal tissue (a and c in Figure 1.A) to difficult cases (i.e., moving them in the area b in Figure 1.C).  
By doing so, we modified the score distributions from A) to C), such that the algorithm could learn key features for classification better by accessing more difficult or challenging cases. 

