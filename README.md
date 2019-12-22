# Max Sliced-Wasserstein Autoencoder - PyTorch


Implementation of Max Sliced Wasserstein Distance in the paper ["Generalized Sliced Wasserstein Distances"](https://arxiv.org/abs/1902.00434) using PyTorch.

## Declaration

This repo is based on the implementation shared by [Emmanuel Fuentes](https://github.com/eifuentes/swae-pytorch), here I only modified the way of obtaining theta.

## Requirement

To run this demo, please install the required by running: `pip install -r requirements-dev.txt`

## Train the model with different setups

You can train this model with 'max' and 'normal' mode, which means using the Maximum Sliced-Wasserstein distance and the normal Sliced-Wasserstein distance, respectively. To train with 'max' mode please run: ` python examples/mnist.py --mode 'max' --mode_test 'max' `. For more informations, please refer to this file: [mnist.py](https://github.com/ShwanMario/max_sliced_wasserstein_distance/blob/master/examples/mnist.py)

