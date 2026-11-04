"""Enregistre des demonstrations de l'expert scripte dans demos.npz.

    python record_demos.py            -> 200 episodes
    python record_demos.py 500        -> 500 episodes

Les seeds partent de 1000 : les episodes d'evaluation (seeds 0..49) ne sont
jamais vus pendant l'entrainement.
Seuls les episodes reussis sont gardes.
"""
import os
import sys

# les scripts partages (environnement, expert) sont dans ../common ; les fichiers produits
# (demos, modeles) sont ecrits a cote de ce script, quel que soit le dossier d'ou on le lance
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)


import numpy as np

from panda_pick_env import PandaPickEnv
from test_expert import expert_action

SEED_OFFSET = 1000


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    env = PandaPickEnv()
    all_obs, all_act, all_ep = [], [], []
    kept = 0
    for ep in range(n_ep):
        obs, _ = env.reset(seed=SEED_OFFSET + ep)
        state = {"phase": "above"}
        ep_obs, ep_act = [], []
        done, success = False, False
        while not done:
            act = expert_action(obs, state)
            ep_obs.append(obs.copy())
            ep_act.append(act.copy())
            obs, _, term, trunc, info = env.step(act)
            success = info["is_success"]   # juge a la FIN : le cube doit etre reste sur la cible
            done = term or trunc
        if success:
            all_obs += ep_obs
            all_act += ep_act
            all_ep += [kept] * len(ep_obs)
            kept += 1
        if (ep + 1) % 20 == 0:
            print(f"{ep + 1}/{n_ep} episodes joues, {kept} gardes")
    env.close()

    np.savez("demos.npz", obs=np.array(all_obs, dtype=np.float32),
             act=np.array(all_act, dtype=np.float32), episode=np.array(all_ep))
    print(f"\ndemos.npz ecrit : {kept} episodes, {len(all_obs)} paires (observation, action)")


if __name__ == "__main__":
    main()
