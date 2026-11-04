"""Diagnostic du BC vision : OU est-ce que ca casse ?

On profite de la simulation : le diagnostic a le droit de lire l'etat exact (le modele, lui, non).
Quatre tests, chacun repond a une question :

  1. Imitation par axe (demos de validation) : le modele imite-t-il moins bien x, y, z ou la pince ?
  2. Sonde de perception : l'image contient-elle assez d'info pour situer le cube et la cible ?
     (on gele le CNN et on lui demande de predire leurs positions ; erreur en mm)
  3. Autopsie des episodes : prise ratee, cube lache en route, ou pose a cote ? de combien de mm ?
  4. Derive en boucle fermee : quand le modele joue seul, s'ecarte-t-il des situations des demos ?
     (on compare son action a celle d'un expert "fantome" qui regarde la meme scene)

    python diagnose_bc_vision.py          -> 20 episodes (environ 2-3 min)
    python diagnose_bc_vision.py 50       -> 50 episodes
    python diagnose_bc_vision.py --model bc_dagger.pt   -> diagnostique un autre modele
Images des echecs (camera fixe | camera poignet) dans le dossier diag/ si opencv est installe.
"""
import os
import sys

# les scripts partages (environnement, expert) sont dans ../common ; les fichiers produits
# (demos, modeles) sont ecrits a cote de ce script, quel que soit le dossier d'ou on le lance
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)


import numpy as np
import torch

from panda_pick_env import CUBE_HALF, SUCCESS_DIST, PandaPickEnv
from test_expert import expert_action
from train_bc_vision import act_from_env, load_policy

torch.manual_seed(0)
N_EP = next((int(a) for a in sys.argv[1:] if a.isdigit()), 20)
N_PROBE = int(os.environ.get("N_PROBE", 600))
AXES = ["x", "y", "z", "pince"]


def save_img(name, img):
    try:
        import cv2
    except ImportError:
        return
    os.makedirs("diag", exist_ok=True)
    big = cv2.resize(img[:, :, ::-1], (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join("diag", name), big)


def features(policy, imgs):
    with torch.no_grad():
        out = []
        for i in range(0, len(imgs), 256):
            x = torch.as_tensor(imgs[i:i + 256]).permute(0, 3, 1, 2).float() / 255.0
            out.append(policy.cnn(x).numpy())
    return np.concatenate(out)


def test_imitation(policy):
    print("\n=== 1. Imitation par axe (demos de validation) ===")
    if not os.path.exists("demos_vision.npz"):
        print("demos_vision.npz absent, test saute")
        return None
    d = np.load("demos_vision.npz")
    ep = d["episode"]
    val = np.isin(ep, np.arange(max(1, (ep.max() + 1) // 10)))       # meme decoupage que train_bc_vision
    img, wr, prop, act = d["img"][val], d["wrist"][val], d["proprio"][val], d["act"][val]
    T = torch.as_tensor
    with torch.no_grad():
        pred = np.concatenate([policy(T(img[i:i + 512]), T(wr[i:i + 512]), T(prop[i:i + 512])).numpy()
                               for i in range(0, len(img), 512)])
    mse = ((pred - act) ** 2).mean(0)
    for a, m in zip(AXES, mse):
        print(f"  {a:6s} erreur quadratique moyenne : {m:.4f}")
    grip_ok = np.mean(np.sign(pred[:, 3]) == np.sign(act[:, 3])) * 100
    print(f"  pince : bonne decision (ouvrir/fermer) dans {grip_ok:.1f} % des pas")
    return mse[:3].mean()


def test_perception(policy, env):
    print(f"\n=== 2. Sonde de perception ({N_PROBE} images camera fixe, bras en position home, CNN gele) ===")
    imgs, labels = [], []
    for k in range(N_PROBE):
        obs, _ = env.reset(seed=5000 + k)
        imgs.append(env.camera_image())
        labels.append([obs[7], obs[8], obs[13], obs[14]])      # cube x, y ; cible x, y
    F = features(policy, np.array(imgs))
    Y = np.array(labels)
    n_tr = int(0.75 * len(F))
    Fm, Fs = F[:n_tr].mean(0), F[:n_tr].std(0) + 1e-6
    X = np.hstack([(F - Fm) / Fs, np.ones((len(F), 1))])
    # regression lineaire regularisee (ridge), solution exacte
    W = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + 10.0 * np.eye(X.shape[1]), X[:n_tr].T @ Y[:n_tr])
    err = np.abs(X[n_tr:] @ W - Y[n_tr:]) * 1000
    names = ["cube x", "cube y", "cible x", "cible y"]
    for i, nm in enumerate(names):
        hasard = np.abs(Y[n_tr:, i] - Y[:n_tr, i].mean()).mean() * 1000
        print(f"  {nm:8s} erreur moyenne {err[:, i].mean():5.1f} mm   (au hasard : {hasard:5.1f} mm)")
    return err.mean()


def test_rollouts(policy, env):
    print(f"\n=== 3 et 4. Autopsie de {N_EP} episodes joues par le modele ===")
    cats = {k: 0 for k in ["reussi", "jamais ferme", "prise ratee", "lache en route", "pas relache", "pose a cote"]}
    grasp_off, final_d, onpol_err, grip_agree, hesit = [], [], [], [], []
    for e in range(N_EP):
        obs, _ = env.reset(seed=e)
        shadow = {}
        max_h, closed_at, grasp_img, ep_err, near_cube = 0.0, None, None, [], 0
        for t in range(100):
            act, frame, wrist = act_from_env(policy, env)
            exp = expert_action(obs, shadow)                  # ce qu'aurait fait l'expert ici
            ep_err.append(((act[:3] - exp[:3]) ** 2).mean())
            grip_agree.append(np.sign(act[3]) == np.sign(exp[3]))
            hesit.append(abs(act[3]) < 0.5)                   # sortie pince ni franchement +1 ni -1
            if np.linalg.norm(obs[10:12]) < 0.015 and abs(obs[12]) < 0.015:
                near_cube += 1                                # outil a hauteur du cube, aligne
            if closed_at is None and act[3] < 0:              # premiere fermeture demandee
                closed_at = t
                grasp_off.append(np.linalg.norm(obs[10:12]) * 1000)   # ecart horizontal outil-cube
                grasp_img = np.concatenate([frame, wrist], axis=1)
            obs, _, _, _, info = env.step(act)
            max_h = max(max_h, obs[9] - CUBE_HALF)
        d = np.linalg.norm(obs[16:18]) * 1000                 # cube - cible a la fin (horizontal)
        width = obs[6]
        held = np.linalg.norm(obs[10:12]) < 0.015 and abs(obs[12]) < 0.015 and 0.03 < width < 0.045
        final_d.append(d)
        onpol_err.append(np.mean(ep_err))
        if info["is_success"]:
            cat = "reussi"
        elif closed_at is None:
            cat = "jamais ferme"
        elif max_h < 0.02:
            cat = "prise ratee"
        elif held:
            cat = "pas relache"
        elif obs[9] - CUBE_HALF > 0.005 or d > 60:
            cat = "lache en route"
        else:
            cat = "pose a cote"
        cats[cat] += 1
        off = f"{grasp_off[-1]:5.1f} mm" if closed_at is not None else "  jamais"
        print(f"  episode {e:2d} : {cat:14s} ecart a la prise {off}   cube-cible fin {d:5.1f} mm"
              f"   pas aligne au cube : {near_cube:3d}")
        if cat != "reussi":
            if grasp_img is not None:
                save_img(f"ep{e:02d}_prise.png", grasp_img)
            save_img(f"ep{e:02d}_fin.png", np.concatenate([env.camera_image(), env.wrist_image()], axis=1))
    return cats, np.array(grasp_off), np.array(final_d), np.mean(onpol_err), np.mean(grip_agree) * 100, \
        np.mean(hesit) * 100


def main():
    policy = load_policy()          # --model fichier.pt, sinon bc_vision.pt
    policy.eval()
    env = PandaPickEnv()
    val_err = test_imitation(policy)
    probe_err = test_perception(policy, env)
    cats, g_off, f_d, onpol, grip, hes = test_rollouts(policy, env)
    env.close()

    print("\n=== Resume ===")
    print("  " + ", ".join(f"{k} : {v}" for k, v in cats.items()))
    if len(g_off):
        print(f"  ecart horizontal outil-cube au moment de fermer : median {np.median(g_off):.1f} mm "
              f"(tolerance : environ 20 mm le long des doigts)")
    print(f"  distance finale cube-cible : mediane {np.median(f_d):.1f} mm (seuil de reussite {SUCCESS_DIST*1000:.0f} mm)")
    print(f"  sonde de perception : erreur moyenne {probe_err:.1f} mm")
    if val_err is not None:
        print(f"  erreur d'action x/y/z : demos de validation {val_err:.4f}  ->  en jouant seul {onpol:.4f}"
              f"  (x{onpol / val_err:.1f})")
    print(f"  pince d'accord avec l'expert fantome : {grip:.1f} % des pas")
    print(f"  pince hesitante (sortie entre -0.5 et 0.5) : {hes:.1f} % des pas")
    print("\nLecture :")
    print("  - sonde > ~10 mm : le CNN localise mal -> probleme de perception (camera, resolution, architecture)")
    print("  - sonde bonne mais erreur 'en jouant seul' >> 'validation' : derive en boucle fermee -> DAgger")
    print("  - prises ratees de peu (ecart 10-30 mm) : precision fine -> camera poignet, resolution")
    print("  - 'jamais ferme' avec beaucoup de pas alignes, ou 'pas relache', + pince hesitante elevee :")
    print("    le modele n'arrive pas a trancher ouvrir/fermer -> seuils de decision de l'expert trop fins")
    if os.path.isdir("diag"):
        print("  Images des echecs (ce que voyait le modele) : dossier diag/")


if __name__ == "__main__":
    main()
