import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self,dim_entree:int, dim_sortie:int):
        """Initialisation des parametres ...
        """
        super(QNN, self).__init__()
        
        "*** TODO ***"
        
    def forward(self, etat: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float)
            
        "*** TODO ***"
        
        return etat


