"""Enregistre des demos de l'expert AVEC les images camera, dans demos_vision.npz.

L'expert lit toujours l'etat exact (position du cube, de la cible).
On enregistre ce que l'ELEVE aura le droit de voir : image camera fixe + image camera poignet
+ proprio (angles + pince),
et l'action de l'expert a imiter.

    python record_demos_vision.py          -> 150 episodes
    python record_demos_vision.py 300      -> 300 episodes (environ 1 Go de RAM par tranche de 100)
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

SEED_OFFSET = 1000      # evaluation sur les seeds 0..49 : jamais vues a l'entrainement


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    env = PandaPickEnv()
    imgs, wrists, props, acts, eps = [], [], [], [], []
    kept = 0
    for ep in range(n_ep):
        obs, _ = env.reset(seed=SEED_OFFSET + ep)
        state = {"phase": "above"}
        e_img, e_wr, e_prop, e_act = [], [], [], []
        done, success = False, False
        while not done:
            act = expert_action(obs, state)
            e_img.append(env.camera_image())
            e_wr.append(env.wrist_image())
            e_prop.append(env.proprio())
            e_act.append(act.copy())
            obs, _, term, trunc, info = env.step(act)
            success = info["is_success"]       # juge a la FIN : le cube doit etre reste sur la cible
            done = term or trunc
        if success:
            imgs += e_img
            wrists += e_wr
            props += e_prop
            acts += e_act
            eps += [kept] * len(e_img)
            kept += 1
        if (ep + 1) % 10 == 0:
            print(f"{ep + 1}/{n_ep} episodes joues, {kept} gardes")
    env.close()

    print("Sauvegarde (compression, patiente un peu)...")
    np.savez_compressed("demos_vision.npz", img=np.array(imgs, dtype=np.uint8),
                        wrist=np.array(wrists, dtype=np.uint8),
                        proprio=np.array(props, dtype=np.float32),
                        act=np.array(acts, dtype=np.float32), episode=np.array(eps))
    print(f"demos_vision.npz ecrit : {kept} episodes, {len(imgs)} pas (2 images par pas)")


if __name__ == "__main__":
    main()
