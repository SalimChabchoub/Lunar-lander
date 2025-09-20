import numpy as np
import random
import torch

from QNN import QNN

class AgentSimple():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self,dim_etat:int = 8, dim_action:int = 4):
        """

        """
        self.dim_etat = dim_etat
        self.dim_action = dim_action
        self.qnn = QNN(dim_etat, dim_action)

    def action_egreedy(self, etat : np.ndarray , eps: float = 0.0) -> int:
        """
            eps: probabilité d'exploration
        """
        if random.random() > eps:
            with torch.no_grad():
                q_values = self.qnn.forward(etat)
                return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.dim_action - 1)
        return 0
    
        



