"""Behavior cloning : un MLP apprend a imiter l'expert a partir de demos.npz,
puis il est evalue seul dans l'environnement sur des positions jamais vues.

    python train_bc.py                 -> entraine, sauve bc_policy.pt, evalue sur 50 episodes
    python train_bc.py --eval          -> evalue bc_policy.pt sans re-entrainer
    python train_bc.py --eval --view   -> 5 episodes avec le viewer
"""
import os
import sys

# les scripts partages (environnement, expert) sont dans ../common ; les fichiers produits
# (demos, modeles) sont ecrits a cote de ce script, quel que soit le dossier d'ou on le lance
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import time

import numpy as np
import torch
import torch.nn as nn

from panda_pick_env import PandaPickEnv

EPOCHS = 60
BATCH = 256
LR = 1e-3
HIDDEN = 256
torch.manual_seed(0)


class Policy(nn.Module):
    """Observation (19) -> action (4). Les observations sont normalisees a l'entree."""

    def __init__(self, obs_mean, obs_std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(obs_mean))
        self.register_buffer("std", torch.as_tensor(obs_std))
        self.net = nn.Sequential(
            nn.Linear(19, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 4), nn.Tanh(),       # actions dans [-1, 1]
        )

    def forward(self, obs):
        return self.net((obs - self.mean) / self.std)


def train():
    data = np.load("demos.npz")
    obs, act, ep = data["obs"], data["act"], data["episode"]
    n_ep = ep.max() + 1
    print(f"{n_ep} episodes, {len(obs)} paires (observation, action)")

    # 10 % des episodes mis de cote pour la validation
    val_ep = np.arange(n_ep)[: max(1, n_ep // 10)]
    val = np.isin(ep, val_ep)
    mean, std = obs[~val].mean(0), obs[~val].std(0) + 1e-6

    X, Y = torch.as_tensor(obs[~val]), torch.as_tensor(act[~val])
    Xv, Yv = torch.as_tensor(obs[val]), torch.as_tensor(act[val])
    policy = Policy(mean, std)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            loss = ((policy(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = ((policy(Xv) - Yv) ** 2).mean().item()
            print(f"epoch {epoch:3d}  loss entrainement {total / len(X):.4f}  loss validation {val_loss:.4f}")

    torch.save(policy.state_dict(), "bc_policy.pt")
    print("Modele sauve dans bc_policy.pt")
    return policy


def load_policy():
    sd = torch.load("bc_policy.pt")
    policy = Policy(sd["mean"].numpy(), sd["std"].numpy())
    policy.load_state_dict(sd)
    return policy


def evaluate(policy, view=False):
    n_ep = 5 if view else 50
    env = PandaPickEnv(render_mode="human" if view else None)
    policy.eval()
    successes = 0
    for e in range(n_ep):
        obs, _ = env.reset(seed=e)              # seeds 0..49 : jamais vus a l'entrainement
        done, success = False, False
        while not done:
            with torch.no_grad():
                act = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            obs, _, term, trunc, info = env.step(act)
            success = info["is_success"]   # juge a la FIN : le cube doit etre reste sur la cible
            done = term or trunc
            if view:
                time.sleep(0.04)
        successes += success
        print(f"episode {e:2d} : {'REUSSI' if success else 'echec'}")
    print(f"\nTaux de reussite du modele BC : {successes}/{n_ep}")
    env.close()


if __name__ == "__main__":
    policy = load_policy() if "--eval" in sys.argv else train()
    evaluate(policy, view="--view" in sys.argv)
