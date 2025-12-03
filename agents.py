import numpy as np
import random
from typing import Dict, Tuple, Any, List
from corridor import Corridor, Action
from collections import defaultdict

# =======================
# 1) Interface d'agent
# =======================

class BaseAgent:
    """Interface minimale : implémente select_action(env, obs)."""
    def __init__(self, name: str = "BaseAgent", seed: int | None = None):
        self.name = name
        if seed is not None:
            random.seed(seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    """Agent aléatoire : choisit uniformément une action légale."""
    def __init__(self, name: str = "RandomAgent", seed: int | None = None):
        super().__init__(name=name, seed=seed)

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        if not actions:
            # Devrait être impossible, mais on sécurise
            raise RuntimeError("Aucune action légale disponible.")
        return random.choice(actions)
    
# Optionnel : exemple d'agent très simple basé sur une heuristique
class GreedyPathAgent(BaseAgent):
    """
    Heuristique: privilégie les déplacements qui rapprochent le pion de sa ligne but.
    Ne place des murs que très rarement (ou jamais).
    """
    def __init__(self, name: str = "GreedyPathAgent", wall_prob: float = 0.0, seed: int | None = None):
        super().__init__(name=name, seed=seed)
        self.wall_prob = wall_prob

    def select_action(self, env: Corridor, obs: Dict) -> Action:
        actions = env.legal_actions()
        # Filtrer les déplacements
        move_actions = [(a, dst) for (a, dst) in actions if a == "M"]
        if not move_actions:
            # Si aucun déplacement légal, choisir un mur légal (si présent)
            return random.choice(actions)

        me = 1 if obs["to_play"] == 1 else 2
        target_row = env.N - 1 if me == 1 else 0

        # Choisir le move qui minimise la distance (en ligne) vers la ligne cible
        def score_move(dst: Tuple[int, int]) -> int:
            r, c = dst
            return abs(target_row - r)

        best = min(move_actions, key=lambda x: score_move(x[1]))
        # Optionnel: parfois poser un mur
        if self.wall_prob > 0 and random.random() < self.wall_prob:
            wall_actions = [(a, w) for (a, w) in actions if a == "W"]
            if wall_actions:
                return random.choice([("W", w) for (_, w) in wall_actions])

        return ("M", best[1])
    
class SARSAAgent(BaseAgent):
    def __init__(self,encoding_function, name:str = "SARSAAgent", seed: int | None = None, gamma: float = 0.99, alpha: float = 0.001, epsilon: float = 1.0,epsilon_decay: float = 0.995, min_epsilon:float = 0.05):
        super().__init__(name=name,seed=seed)
        self.n_actions = 7
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(self.n_actions)
        self.q_table = defaultdict(lambda:np.zeros(self.n_actions,dtype=float))
        self.encoding_state = encoding_function
    def get_strategic_walls(self,env:Corridor,obs:Dict,radius:int=2)->List[Action]:
        """ Retourne trois actions de type 'wall' qui cerne le pion adverse"""
        player = int(obs['to_play'])
        opp = env.opponent_pawn(player)
        orow, ocol = opp
        legal_walls = set(env.legal_walls(player)) 
        candidates = []

        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                for ori in ("H","V"):
                    r = orow + dr
                    c = ocol + dc
                    if not (0<= r<env.N -1 and 0<=c<env.N-1):
                        continue
                    wall = ("W",(r,c,ori))
                    if wall not in legal_walls:
                        continue
                    dist = abs(dr) + abs(dc)
                    candidates.append((dist,wall))
          
        candidates.sort(key=lambda x:x[0])
        return [wall for _,wall in candidates[:3]]
    def select_action(self,env: Corridor, obs: Dict,state) -> Action:
        if np.random.uniform() < self.epsilon:
            action = random.choice(self.actions)
        else:
            encoded_state = self.encoding_state(state)
            action = np.argmax(self.q_table[encoded_state])
        self.epsilon = max(self.min_epsilon,self.epsilon*self.epsilon_decay)
        return action
    
    def update(self, state, action, reward, next_state, done):
        pass