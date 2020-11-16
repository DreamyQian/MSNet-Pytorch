# MSNet-Pytorch

This repository contains the implementation of the paper (MSNet).

- MSNet: Multi-resolution Synergistic Networks for Adaptive Inference

## Introduction

## Results

- Anytime prediction results on the CIFAR-10, CIFAR-100 and ImageNet


- Budgeted prediction results on the CIFAR-10, CIFAR-100 and ImageNet


- Visualization


## Dependencies:

+ Python3
+ PyTorch >= 1.0

## Get Started

We Provide shell scripts for training a MSNet on CIFAR and ImageNet.

### Train an MSNet (blocks=7, step=4-4-6-2-2-10-2) on CIFAR-100.
+ **Step 1: Training an MSNet**  
    Modify the run_step1.sh to config your path to the dataset, your GPU devices and your saving directory. Then run
    ```sh
    bash run_step1.sh
    ```
    Configurations on CIFAR-100.
    ```sh
    python main_step1.py --data-root /PATH/TO/CIFAR100  --save /PATH/TO/SAVE --data cifar100 \
                         --gpu 0 --arch msnet --batch-size 128 --epochs 310 --lr-type SGDR \
                         --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6 \
                         --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 -j 6
    ```

+ **Step 2: Fine-tune the network with Adaptive loss**    
    Modify the run_step2.sh to config your path to the dataset, your GPU devices and your saving directory (different from the saving directory of Step 1). 
    ```sh
    bash run_step2.sh
    ```

### Train an MSNet (blocks=7, step=4-4-2-6-2-8-2) on ImageNet.
+ **Step 1: Training an MSNet**  
    Modify the run_step1.sh to config your path to the dataset, your GPU devices and your saving directory. Then run
    ```sh
    bash run_step1.sh
    ```
    Configurations on ImageNet.
    ```sh
    python main_step1.py --data-root /PATH/TO/ImageNet  --save /PATH/TO/SAVE --data ImageNet \
                         --gpu 0,1,2,3 --arch msnet --batch-size 256 --epochs 150 --lr-type SGDR \
                         --nBlocks 7 --step 4-4-2-6-2-8-2 --nChannels 32 --growthRate 16 \
                         --grFactor 1-2-4-8 --bnFactor 1-2-4-8 --transFactor 1-3-5 -j 6
    ```

+ **Step 2: Fine-tune the network with Adaptive loss**    
    Modify the run_step2.sh to config your path to the dataset, your GPU devices and your saving directory (different from the saving directory of Step 1). 
    ```sh
    bash run_step2.sh
    ```

### Test an MSNet (blocks=7, step=4-4-6-2-2-10-2) on CIFAR-100.

Modify the run_inference.sh to config your path to the CIFAR-100, your GPU devices and your saving directory. Then run
```sh
bash run_inference.sh
```

Test configurations.
```sh
python main_step1.py --data-root /PATH/TO/CIFAR100  --save /PATH/TO/SAVE --data cifar100 \
                     --gpu 0 --arch msnet --batch-size 128 --epochs 310 --lr-type SGDR \
                     --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6 \
                     --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 --evalmode dynamic \
                     --evaluate-from /PATH/TO/CHECKPOINT/ -j 6
```

## Contact
If you have any question, please feel free to contact the authors.

## Acknowledgment
We use the pytorch implementation of MSDNet in our experiments. The code can be found [here](https://github.com/kalviny/MSDNet-PyTorch).



