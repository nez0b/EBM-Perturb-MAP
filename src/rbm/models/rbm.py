"""
Restricted Boltzmann Machine implementation with Perturb-and-MAP training.

This module contains the core RBM model class that implements the Perturb-and-MAP
methodology for training using QUBO solvers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class RBM(nn.Module):
    """
    A Restricted Boltzmann Machine implemented with PyTorch.
    
    This implementation includes methods for both joint and conditional sampling
    using the Perturb-and-MAP methodology with QUBO solvers.
    
    Args:
        n_visible (int): Number of units in the visible layer.
        n_hidden (int): Number of units in the hidden layer.
    """
    
    def __init__(self, n_visible: int, n_hidden: int):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weight matrix connecting visible and hidden layers
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        # Bias for the hidden layer
        self.c = nn.Parameter(torch.zeros(n_hidden))
        # Bias for the visible layer
        self.b = nn.Parameter(torch.zeros(n_visible))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass from visible to hidden.
        
        Args:
            v: Visible layer input tensor.
            
        Returns:
            Hidden layer activation probabilities.
        """
        h_prob = torch.sigmoid(torch.nn.functional.linear(v, self.W, self.c))
        return h_prob

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """
        Standard backward pass (reconstruction) from hidden to visible.
        
        Args:
            h: Hidden layer input tensor.
            
        Returns:
            Visible layer activation probabilities.
        """
        v_prob = torch.sigmoid(torch.nn.functional.linear(h, self.W.t(), self.b))
        return v_prob

    def create_joint_qubo(self) -> np.ndarray:
        """
        Creates the QUBO matrix for sampling from the JOINT distribution P(v, h)
        using the Perturb-and-MAP method.

        The perturbed energy function to minimize is:
        E'(v, h) = -h^T W v - (c^T - g_h^T)h - (b^T - g_v^T)v
        where g_h and g_v are Gumbel-distributed noise vectors.

        This is converted to the QUBO form: x^T Q x, where x = [v, h].

        Returns:
            The QUBO matrix Q for the joint distribution.
        """
        N = self.n_visible + self.n_hidden
        Q_joint = np.zeros((N, N))

        # --- PERTURB STEP ---
        # Add Gumbel noise to the biases. This is the "perturb" part of P&M.
        # Gumbel(0,1) noise ~ -log(-log(U)) where U ~ Uniform(0,1)
        gumbel_v = -torch.log(-torch.log(torch.rand(self.n_visible)))
        gumbel_h = -torch.log(-torch.log(torch.rand(self.n_hidden)))

        perturbed_b = self.b - gumbel_v
        perturbed_c = self.c - gumbel_h

        # --- MAP to QUBO FORMULATION ---
        # The QUBO is formed from the coefficients of the energy function.
        # For a variable x_i, a linear term 'a*x_i' becomes 'a*x_i^2' in QUBO,
        # so 'a' goes on the diagonal Q_ii.
        # A quadratic term 'b*x_i*x_j' goes on the off-diagonal Q_ij and Q_ji.

        # 1. Diagonal terms from perturbed biases (-b' and -c')
        diag_terms = -torch.cat((perturbed_b, perturbed_c))
        np.fill_diagonal(Q_joint, diag_terms.detach().numpy())

        # 2. Off-diagonal terms from the weight matrix (-W)
        # These terms couple the visible (v) and hidden (h) units.
        # The variables are ordered as x = [v_1..v_n, h_1..h_m].
        # The interaction block is in the top-right and bottom-left of Q.
        W_numpy = self.W.detach().numpy()  # Shape (n_hidden, n_visible)

        # We set Q_ij = Q_ji = -W / 2, but since our solver might expect
        # the full coefficient, we put -W in one of the blocks. The solver
        # should handle the symmetric representation.
        # Note the transpose for correct alignment.
        Q_joint[:self.n_visible, self.n_visible:] = -W_numpy.T
        
        # Symmetrize the matrix for standard QUBO solvers
        Q_joint = (Q_joint + Q_joint.T) / 2.0
        return Q_joint

    def create_qubo_for_sampling(self, v_in: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the QUBO for sampling the hidden layer h given a visible layer v.
        Used for inference/reconstruction.
        
        Args:
            v_in: Input visible layer tensor.
            
        Returns:
            Tuple of (QUBO matrix, linear terms) for the hidden layer sampling.
        """
        if v_in.dim() > 1:
            v_in = v_in.squeeze(0)

        n_hidden = self.W.shape[0]
        h_activations = torch.matmul(self.W, v_in) + self.c
        gumbel_noise = -torch.log(-torch.log(torch.rand(n_hidden)))
        linear_terms = gumbel_noise - h_activations
        Q = np.diag(linear_terms.detach().numpy())
        return Q, linear_terms.detach().numpy()