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
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie
        self.fc1 = nn.Linear(dim_entree, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, dim_sortie)
        

        
    def forward(self, etat: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float)
            
        "*** TODO ***"
        etat = F.relu(self.fc1(etat))
        etat = F.relu(self.fc2(etat))
        etat = F.relu(self.fc3(etat))
        return self.fc4(etat)

