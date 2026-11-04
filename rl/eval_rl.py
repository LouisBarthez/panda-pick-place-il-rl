"""Evalue le modele RL (SAC) sur 50 episodes jamais vus, ou le montre en action.

    python eval_rl.py                  -> 50 episodes, taux de reussite (modele sac_best.zip)
    python eval_rl.py --view           -> 5 episodes avec le viewer
    python eval_rl.py --last           -> utilise sac_last.zip au lieu du meilleur
    python eval_rl.py --plot           -> courbe d'apprentissage (rl_log.csv -> rl_courbe.png)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import time  # noqa: E402

import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from panda_pick_env import PandaPickEnv  # noqa: E402


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.genfromtxt("rl_log.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["pas"] / 1000, d["reussites_sur_20"] / 20 * 100, marker="o")
    ax.set_xlabel("pas d'entrainement (milliers)")
    ax.set_ylabel("taux de reussite (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SAC sur le pick-and-place (evaluation sur 20 episodes)")
    ax.grid(alpha=0.3)
    fig.savefig("rl_courbe.png", dpi=130, bbox_inches="tight")
    print("Courbe enregistree : rl_courbe.png")


def main():
    if "--plot" in sys.argv:
        return plot()
    tag = "last" if "--last" in sys.argv else "best"
    view = "--view" in sys.argv
    venv = VecNormalize.load(f"vecnormalize_{tag}.pkl", DummyVecEnv([PandaPickEnv]))
    venv.training = False
    model = SAC.load(f"sac_{tag}")
    env = PandaPickEnv(render_mode="human" if view else None)
    n_ep, ok = (5 if view else 50), 0
    for e in range(n_ep):
        obs, _ = env.reset(seed=e)                       # seeds 0..49
        for _ in range(100):
            act, _ = model.predict(venv.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
            obs, _, _, _, info = env.step(act[0])
            if view:
                time.sleep(0.04)
        ok += info["is_success"]
        print(f"episode {e:2d} : {'REUSSI' if info['is_success'] else 'echec'}")
    print(f"\nTaux de reussite du modele RL (sac_{tag}) : {ok}/{n_ep}")
    env.close()


if __name__ == "__main__":
    main()
