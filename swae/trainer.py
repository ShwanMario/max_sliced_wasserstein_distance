import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from swae.distributions import rand_cirlce2d
import torch.optim as optim


def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w ** 2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def rand_theta(num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    theta_distribution = torch.distributions.uniform.Uniform(0, 2 * np.pi)
    theta = theta_distribution.sample((num_samples,))
    return theta


def theta_to_coordinates(theta):
    projections = torch.zeros([theta.shape[0], 2])
    projections[:, 0] = torch.cos(theta)
    projections[:, 1] = torch.sin(theta)
    return projections


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu', mode='max'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution

    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space

    if mode == 'max':
        encoded_samples_ = encoded_samples.detach()
        theta = rand_theta(num_samples=1).to(device)
        theta = nn.Parameter(theta)
        projections = theta_to_coordinates(theta).to(device)
        parameters = nn.ParameterList([theta])
        optimizer = optim.Adam(parameters, lr=1e-1)
        criterior = 1e-3
        converge = False
        negative_w_distance = torch.tensor(0)
        while (not converge):
            old_distance = -negative_w_distance.clone()
            optimizer.zero_grad()
            # calculate projections through the encoded samples
            # print('detached_encoded_samples:',encoded_samples.mean().item())
            encoded_projections = encoded_samples_.matmul(projections.transpose(0, 1))
            # calculate projections through the prior distribution random samples
            distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
            # calculate the sliced wasserstein distance by
            # sorting the samples per random projection and
            # calculating the difference between the
            # encoded samples and drawn random samples
            # per random projection
            wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                    torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
            # distance between latent space prior and encoded distributions
            # power of 2 by default for Wasserstein-2
            wasserstein_distance = torch.pow(wasserstein_distance, p)

            negative_w_distance = -wasserstein_distance.mean()
            negative_w_distance.backward(retain_graph=True)
            optimizer.step()
            projections = theta_to_coordinates(theta).to(device)
            # print(torch.abs(old_distance+negative_w_distance).item(),-negative_w_distance.item(),theta.item())
            if (torch.abs(old_distance + negative_w_distance) < criterior):
                converge = True
                theta.detach_()
                # print('converged')
    else:
        theta = rand_theta(num_samples=num_projections).to(device)
        projections = theta_to_coordinates(theta).to(device)
        # projections = rand_projections(embedding_dim, num_projections).to(device)
    # approximate mean wasserstein_distance for each projection

    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)

    # print(wasserstein_distance.mean().item())
    return wasserstein_distance.mean()


'''
plot theta and updated theta while debugging:

import tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(projections[:,0].detach().numpy(),projections[:,1].detach().numpy(),linewidth=0.1,label='updated theta')
plt.scatter(old_projections[:,0].detach().numpy(),old_projections[:,1].detach().numpy(),linewidth=0.1,label='theta')
plt.legend()
plt.show()

'''


def sliced_wasserstein_distance(encoded_samples,
                                distribution_fn=rand_cirlce2d,
                                num_projections=50,
                                p=2,
                                device='cpu', mode='max'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw random samples from latent space prior distribution
    z = distribution_fn(batch_size).to(device)
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, z,
                                       num_projections, p, device, mode=mode)
    return swd


class SWAEBatchTrainer:
    """ Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            weight (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """

    def __init__(self, autoencoder, optimizer, distribution_fn,
                 num_projections=50, p=2, weight=10.0, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight = weight
        self._device = device if device else torch.device('cpu')

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x, mode='max'):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x, mode=mode)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x, mode='max'):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x, mode=mode)
        return evals

    def eval_on_batch(self, x, mode='max'):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        # mutual information reconstruction loss
        bce = F.binary_cross_entropy(recon_x, x)
        # for explaination of additional L1 loss see references in README.md
        # high lvl summary prevents variance collapse on latent variables
        l1 = F.l1_loss(recon_x, x)
        # divergence on transformation plane from X space to Z space to match prior
        _swd = sliced_wasserstein_distance(z, self._distribution_fn,
                                           self.num_projections_, self.p_,
                                           self._device, mode=mode)
        # z.requires_grad=True
        w2 = float(self.weight) * _swd  # approximate wasserstein-2 distance
        loss = bce + l1 + w2
        # print('CrossEntorpy:',bce.item(),'L1Loss:',l1.item(),'W2Loss:',w2.item())
        return {
            'loss': loss,
            'bce': bce,
            'l1': l1,
            'w2': w2,
            'encode': z,
            'decode': recon_x,
            'real_loss': bce + l1 + _swd
        }
