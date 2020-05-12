# Learning SSBM with selfplay and deep learning

See [ssbm_gym](https://github.com/Gurvan/ssbm_gym).


## Infos

This repository is experimental and work in progress. No support will be provided.

It has been tested on Ubuntu 18.04.

This uses a variant of [IMPALA](https://arxiv.org/abs/1802.01561) made for running on a personal computer.

The pretained model is not good at all at the moment.

## Requirements

- pytorch
- numpy
- matplotlib
- ssbm_gym


## Usage

Training:

- `python train.py`
- `python train.py --load-model results/pretrained/model.pth` for resuming training

Playing:

- `python test.py`
- `python test.py --human` for playing against the bot as player 2 if you have a GameCube adapter

