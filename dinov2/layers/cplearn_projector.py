# Copyright 2025 CPLearn team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov2.loss.cplearn_loss import cplearn_loss_func
import numpy as np
import math


class CPLearnProjector(nn.Module):
    """Projector used by the original CPLearn implementation."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.proj_hidden_dim = hidden_dim
        self.proj_output_dim = proj_dim
        self.epsilon = epsilon

        self.projector = nn.Sequential(
            nn.Linear(in_dim, self.proj_hidden_dim),
            #nn.BatchNorm1d(self.proj_hidden_dim),
            nn.LayerNorm(self.proj_hidden_dim)
        )
        self.tanh = nn.Tanh()
        self.register_buffer('weights', torch.tensor(2. * np.random.randint(2, size=(self.proj_hidden_dim, self.proj_output_dim)) - 1.).to(torch.float))
        # weights = 2.0 * np.random.randint(2, size=(hidden_dim, proj_dim)) - 1.0
        # self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.projector.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        weights = 2.0 * np.random.randint(2, size=(self.proj_hidden_dim, self.proj_output_dim)) - 1.0
        with torch.no_grad():
            self.weights.copy_(torch.tensor(weights, dtype=self.weights.dtype, device=self.weights.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.projector[0].weight.dtype)   # match BN/Linear dtype
        z = self.projector(x)
        z = self.tanh(z)
        z = z @ self.weights.to(z.dtype)
        n, c = z.shape
        arg = (1.0 - self.epsilon * (float(c) - 1.0)) / self.epsilon
        arg = max(arg, self.epsilon) 
        temp = float(self.proj_hidden_dim) / (math.sqrt(float(n)) * math.log(arg))
        return z / temp