import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logging = logging.getLogger("dinov2")
class KDELoss(nn.Module):

    def __init__(self):
        super().__init__()

    
    def vmF(self, x, y, kappa):

        x = x.float()
        y = y.float()
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        sim = x_norm @ y_norm.T#Doesn't follow the formula technically, but shouldn't matter here.

        kernel = torch.exp(kappa * sim)
        return kernel
    
    def forward(self, student_output, eps=1e-8):
        
        kappa = 5.0
        #Batch size
        n = student_output.shape[0]
        k = self.vmF(student_output, student_output, kappa)
        density_estimates = k.sum(dim=1)  # [n]
        entropy = -torch.log(density_estimates + 1e-9).mean()
        return entropy
