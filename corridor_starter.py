import random
from typing import Dict, Tuple, Any, List
from corridor import Corridor, Action

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


# =======================
# 2) Boucle de partie
# =======================

def play_game(env: Corridor, agent1: BaseAgent, agent2: BaseAgent, render: bool = False, max_moves: int = 500) -> dict:
    obs = env.reset()
    if render:
        env.render()

    agents = {1: agent1, 2: agent2}
    history: List[dict] = []

    for _ in range(max_moves):
        player = obs["to_play"]
        agent = agents[player]
        action = agent.select_action(env, obs)
        obs, reward, done, info = env.step(action)

        history.append({
            "player": player,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        })

        if render:
            env.render()

        if done:
            winner = info.get("winner")
            return {
                "winner": winner,
                "move_count": env.move_count,
                "history": history
            }

    # Sécurité: match nul si trop long
    return {"winner": None, "move_count": env.move_count, "history": history}


def evaluate(n_games: int = 50, render: bool = False):
    env = Corridor(N=9, walls_per_player=10)

    # Remplace RandomAgent par votre agent :
    agent1 = RandomAgent(name="Random-1", seed=123)
    agent2 = GreedyPathAgent(name="Greedy-2", wall_prob=0.0, seed=321)

    results = {"P1": 0, "P2": 0, "Draw": 0}
    for g in range(n_games):
        # Alterner qui commence
        if g % 2 == 0:
            # P1 = agent1, P2 = agent2
            pass
        else:
            # On échange les noms pour garder affichage cohérent
            agent1, agent2 = agent2, agent1

        out = play_game(env, agent1, agent2, render=render)
        winner = out["winner"]
        if winner == 1:
            results["P1"] += 1
        elif winner == 2:
            results["P2"] += 1
        else:
            results["Draw"] += 1

        # Ré-inverse pour la prochaine itération si on avait inversé
        if g % 2 == 1:
            agent1, agent2 = agent2, agent1

    total = n_games
    print(f"\n=== Evaluation over {total} games ===")
    print(f"P1 wins: {results['P1']}")
    print(f"P2 wins: {results['P2']}")
    print(f"Draws  : {results['Draw']}")
    print("====================================")


if __name__ == "__main__":
    # Lancer une partie unique avec rendu:
    play_game(Corridor(), RandomAgent(), RandomAgent(), render=True)

    # Lancer une évaluation
    # evaluate(n_games=20, render=False)