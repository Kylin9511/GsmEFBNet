## Overview
This is the PyTorch implementation of paper [Deep Learning for Hybrid Beamforming with Finite Feedback in GSM Aided mmWave MIMO Systems](arxiv_wip) which has been submitted to IEEE for possible publication. The general spatial modulation scheme is first considered into the joint optimized pipeline of channel estimation, CSI feedback, and beamforming. The test script and trained models are listed here and the key results can be reproduced as a validation of our work.

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.8
- [PyTorch >= 1.10.1](https://pytorch.org/get-started/locally/)
- [fvcore](https://github.com/facebookresearch/fvcore)


## Project Preparation

#### A. Data Preparation
The channel state information (CSI) matrix is generated according to the influential clustered Saleh Valenzuela (SV) model. The test dataset is provided [Baidu Netdisk](https://pan.baidu.com/s/1WAdWAe10t5cmlq9M9hu1wg)(passwd: 842t), which is easy for you to download and reproduce the experiment results.
You can also generate your own dataset according to the SV channel model. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading
The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/11Ml_UlQF2DJZkK95yoEMTQ)(passwd: wh9v).

#### C. Project Tree Arrangement
We recommend you to arrange the project tree as follows.

```
home
├── GsmEFBNet  # The cloned GsmEFBNet repository
│   ├── dataset
│   ├── model
│   ├── utils
│   ├── main.py
├── data_files # Test data files
│   ├── Data_Nt16Nr4Nrf2Ncl2Nray8_2000.mat
├── Nt16Nr4Nrf2Nk4      # Checkpoints under the scenario (Nt=16,Nr=4,Nrf=2,Nk=4)
│   ├── AblationOnB  # Checkpoints for the ablation study vs. B (SNR=10dB)
│   │     ├── model_B3.pth
│   │     ├── ...
│   ├── AblationOnSNRWithB6 # Checkpoints for the ablation study vs. SNR (B=6)
│   │     ├── model_SNR-3.pth
│   │     ├── ...
│   ├── AblationOnSNRWithB36 # Checkpoints for the ablation study vs. SNR (B=36)
│   │     ├── model_SNR-3.pth
│   │     ├── ...
├── evaluate.sh  # The test script
...
```

## Results and Reproduction

The main results of the deep learning method reported in our paper are presented in the following tables. All the listed results are marked in Fig. 3 and Fig. 4 in our paper. Our proposed GsmEFBNet first introduces the general spatial modulation (GSM) scheme into the end-to-end jointly optimization pipeline of channel estimation, CSI feedback and beamforming.

The performance and corresponding checkpoints of Fig. 3 in the paper is given as follows.
Feedback Bits B | Sum Rate (Bits/s/Hz) | Checkpoint
 :--: | :--: | :--:
3 | 8.5679 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B3.pth
10 | 8.8024 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B10.pth
20 | 9.5025 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B20.pth
30 | 10.0396 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B30.pth
40 | 10.0978 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B40.pth
50 | 10.1071 | Nt16Nr4Nrf2Nk4/AblationOnB/model_B50.pth

The performance and corresponding checkpoints of Fig. 4 in the paper is given as follows.
Feedback Bits B | SNR (dB) | Sum Rate (Bits/s/Hz) | Checkpoint
 :--: | :--: | :--: | :--:
6 | -10 | 0.5604 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR-10.pth
6 | -7 | 1.0139 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR-7.pth
6 | -5 | 1.4414 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR-5.pth
6 | -3 | 1.9514 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR-3.pth
6 | 0 | 3.2281 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR0.pth
6 | 3 | 4.4839 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR3.pth
6 | 5 | 5.5980 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR5.pth
6 | 7 | 6.9675 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR7.pth
6 | 10 | 8.6310 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR10.pth
6 | 13 | 10.8503 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR13.pth
6 | 15 | 12.3268 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB6/model_SNR15.pth
36 | -10 | 0.6634 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR-10.pth
36 | -7 | 1.2979 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR-7.pth
36 | -5 | 1.9451 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR-5.pth
36 | -3 | 2.6870 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR-3.pth
36 | 0 | 4.1003 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR0.pth
36 | 3 | 5.6763 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR3.pth
36 | 5 | 6.8352 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR5.pth
36 | 7 | 8.1042 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR7.pth
36 | 10 | 10.0434 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR10.pth
36 | 13 | 12.1352 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR13.pth
36 | 15 | 13.5398 | Nt16Nr4Nrf2Nk4/AblationOnSNRWithB36/model_SNR15.pth

As aforementioned, we provide model checkpoints for all the deep learning-based results. Our code library supports easy inference. *It is worth mentioning that the inference results have a certain degree of randomness brought by the random Gaussian noise in the SubArrayGSMPilotNet.*

**To reproduce all these results, you need to download the given dataset and corresponding checkpoints. Also, you should arrange your projects tree as instructed.** An example of `evaluate.sh` is shown as follows.
Change the SNR and feedback bits with `SNR` and `B`, respectively. Change the number of transmit antennas, receive antennas, RF chains and users with `Nt`, `Nr`, `Nrf` and `Nk`, respectively. Change the pilot length with `L`.

``` bash
python3 home/GsmEFBNet/main.py \
    --evaluate \
    --pretrained home/Nt8Nr2Nrf1Nk4/AblationOnB/model_B30.pth \
    --gpu 0 \
    --test-data-dir home/data_files/Data_Nt8Nr2Nrf1Ncl2Nray8_2000.mat \
    --model GsmEFBNet \
    -SNR 10 \
    -B 30 \
    -Nt 8 \
    -Nr 2 \
    -Nrf 1 \
    -Nk 4 \
    -TP 1 \
    -L 4 \
    2>&1 | tee log_inference.txt
```

## Acknowledgment
This repository is constructed referring to [DL-DSC-FDD](https://github.com/foadsohrabi/DL-DSC-FDD-Massive-MIMO). Thank Foad Sohrabi, Kareem M. Attiah, and Wei Yu for their excellent work and you can find it in detail from this [paper](https://ieeexplore.ieee.org/document/9347820).
