"""DAgger (Dataset Aggregation) pour le BC vision.

Principe, a chaque iteration :
  1. le MODELE joue seul des episodes (c'est toujours son action qui est executee) ;
  2. a chaque pas, on demande a l'EXPERT ce qu'il aurait fait dans cette situation
     (il lit l'etat exact) et on enregistre : 2 images + proprio + action de l'expert ;
  3. on ajoute ces exemples aux demos et on re-entraine le modele ;
  4. on evalue le nouveau modele.
Le modele apprend ainsi a se rattraper dans les situations ou SES erreurs l'amenent,
situations que l'expert seul ne visite jamais.

Il faut avoir lance avant : record_demos_vision.py puis train_bc_vision.py (avec l'expert actuel).

    python dagger.py          -> 3 iterations
    python dagger.py 5        -> 5 iterations
Sorties : bc_dagger.pt (dernier modele) + bc_dagger_it1.pt, it2... (un par iteration)
Revoir : python train_bc_vision.py --eval --view --model bc_dagger.pt
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

from panda_pick_env import PandaPickEnv
from test_expert import expert_action
from train_bc_vision import BATCH, LR, act_from_env, batched_loss, load_policy, random_shift

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_ROLL = 30          # episodes joues par le modele a chaque iteration
FT_EPOCHS = 40       # epochs de re-entrainement a chaque iteration (15 etait trop court le 31/10 :
                     # le modele n'avait pas le temps d'apprendre le dataset agrandi)
N_EVAL = 30          # episodes d'evaluation a chaque iteration (seeds 0..29)
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 4)


def collect(policy, env, it):
    imgs, wrists, props, labels = [], [], [], []
    for k in range(N_ROLL):
        obs, _ = env.reset(seed=20000 + 1000 * it + k)    # positions jamais vues ailleurs
        st = {}
        for _ in range(100):
            prop = env.proprio()
            act, front, wrist = act_from_env(policy, env)
            imgs.append(front)
            wrists.append(wrist)
            props.append(prop)
            labels.append(expert_action(obs, st))          # ce que l'expert aurait fait ICI
            obs, *_ = env.step(act)                        # mais c'est le modele qui agit
    return (np.array(imgs), np.array(wrists), np.array(props, dtype=np.float32),
            np.array(labels, dtype=np.float32))


def finetune(policy, I, W, P, A):
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    I, W, P, A = torch.as_tensor(I), torch.as_tensor(W), torch.as_tensor(P), torch.as_tensor(A)
    for epoch in range(1, FT_EPOCHS + 1):
        t0 = time.time()
        policy.train()
        perm = torch.randperm(len(I))
        total = 0.0
        for i in range(0, len(I), BATCH):
            idx = perm[i:i + BATCH]
            loss = ((policy(random_shift(I[idx]), random_shift(W[idx]), P[idx]) - A[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"    epoch {epoch:2d}  loss {total / len(I):.4f}  ({time.time() - t0:.0f} s)")
    policy.eval()


def evaluate(policy, env):
    ok = 0
    for e in range(N_EVAL):
        env.reset(seed=e)
        for _ in range(100):
            _, _, _, _, info = env.step(act_from_env(policy, env)[0])
        ok += info["is_success"]
    return ok


def main():
    d = np.load("demos_vision.npz")
    I, W, P, A = d["img"], d["wrist"], d["proprio"], d["act"]
    print(f"Demos de depart : {len(I)} pas")
    policy = load_policy("bc_vision.pt")
    policy.eval()
    T = torch.as_tensor
    print(f"Loss du modele de depart sur ses demos : {batched_loss(policy, T(I), T(W), T(P), T(A)):.4f}"
          "  (le re-entrainement doit revenir a ce niveau)")
    env = PandaPickEnv()
    print(f"Modele de depart : {evaluate(policy, env)}/{N_EVAL}")

    for it in range(1, N_ITER + 1):
        t0 = time.time()
        print(f"\n=== Iteration {it}/{N_ITER} ===")
        print(f"  1. le modele joue {N_ROLL} episodes, l'expert annote chaque pas...")
        ni, nw, npr, na = collect(policy, env, it)
        I, W = np.concatenate([I, ni]), np.concatenate([W, nw])
        P, A = np.concatenate([P, npr]), np.concatenate([A, na])
        print(f"  2. dataset agrege : {len(I)} pas")
        print(f"  3. re-entrainement ({FT_EPOCHS} epochs)")
        finetune(policy, I, W, P, A)
        torch.save(policy.state_dict(), f"bc_dagger_it{it}.pt")
        torch.save(policy.state_dict(), "bc_dagger.pt")
        print(f"  4. evaluation : {evaluate(policy, env)}/{N_EVAL}   ({(time.time() - t0) / 60:.1f} min)")
    env.close()
    print("\nModele final : bc_dagger.pt")
    print("Revoir : python train_bc_vision.py --eval --view --model bc_dagger.pt")
    print("Diagnostic : python diagnose_bc_vision.py --model bc_dagger.pt")


if __name__ == "__main__":
    main()
