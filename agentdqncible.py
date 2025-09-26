import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNCible():
    """Agent qui utilise l'algorithme DQN avec réseau cible."""

    def __init__(self, dim_etat:int, dim_action:int, gamma=0.99):
        """Constructeur.
        

        """
        self.replaybuffer = ReplayBuffer(100000, 64)
        self.dim_action = dim_action
        self.dim_etat = dim_etat
        self.gamma = gamma
        self.state_size = dim_etat
        self.action_size = dim_action
        self.qnn = QNN(dim_etat, dim_action)
        self.optimizer = optim.Adam(self.qnn.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.qnn_cible = QNN(dim_etat, dim_action)
        self.qnn_cible.load_state_dict(self.qnn.state_dict())

        self.learn_step = 0      # Compteur de steps
        self.C = 100               # Fréquence d’update du target network
        

    def phase_echantillonage(self,etat : np.ndarray ,action : np.ndarray ,recompense: float,etat_suivant: np.ndarray ,terminaison: bool):
        self.replaybuffer.add(etat, action, recompense, etat_suivant, terminaison)
        return 0
        
    def phase_apprentissage(self):
        sj,aj,rj,sJplus,dones = self.replaybuffer.sample()
        q_values = self.qnn.forward(sj).gather(1, aj)
        with torch.no_grad():
            q_next = self.qnn_cible.forward(sJplus).max(1)[0].unsqueeze(1)
            q_target = rj + (self.gamma * q_next * (1 - dones))
        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.C == 0:
            for param_duplicat, param_source in zip(self.qnn_cible.parameters(), self.qnn.parameters()):
                param_duplicat.data.copy_(param_source.data)
        return 0
    
    
    def action_egreedy(self, etat : np.ndarray ,eps: float = 0.0) -> int:
        if random.random() > eps:
            with torch.no_grad():
                q_values = self.qnn.forward(etat)
                return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.dim_action - 1)
        return 0