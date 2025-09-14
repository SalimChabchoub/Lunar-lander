import numpy as np
import random
import torch

from QNN import QNN

class AgentSimple():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self):
        """

        """

    def action_egreedy(self, etat : np.ndarray , eps: float = 0.0) -> int:
        """
            eps: probabilité d'exploration
        """
        
        return 0
    
        



