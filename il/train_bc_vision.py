"""BC vision : le modele ne voit QUE deux images (camera fixe + camera poignet) et la proprio
(7 angles articulaires + pince). Il n'a jamais acces a la position du cube ni de la cible.

    python train_bc_vision.py                 -> entraine, sauve bc_vision.pt, evalue sur 50 episodes
    python train_bc_vision.py --eval          -> evalue bc_vision.pt sans re-entrainer
    python train_bc_vision.py --eval --model bc_dagger.pt   -> evalue un autre modele
    python train_bc_vision.py --eval --view   -> 5 episodes avec le viewer + une fenetre
                                                 "ce que voit le modele" (les 2 cameras agrandies)
    Pour la fenetre camera : python -m pip install opencv-python
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
import torch.nn.functional as F

from panda_pick_env import IMG_SIZE, PandaPickEnv

EPOCHS = 80          # la loss baissait encore a 25 epochs : on laisse le modele converger
BATCH = 128
LR = 3e-4
SHIFT = 4            # augmentation : decalage aleatoire de l'image de +-4 pixels
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 4)


def make_cnn():
    """Image 3 x 96 x 96 -> 256 caracteristiques."""
    return nn.Sequential(                                         # 3 x 96 x 96
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),      # 32 x 48 x 48
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),     # 32 x 24 x 24
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),     # 64 x 12 x 12
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),     # 64 x 6 x 6
        nn.Flatten(),
        nn.Linear(64 * (IMG_SIZE // 16) ** 2, 256), nn.ReLU(),
    )


class VisionPolicy(nn.Module):
    """Un CNN par camera (poids separes) + MLP (features des 2 cameras + proprio -> action)."""

    def __init__(self, prop_mean, prop_std):
        super().__init__()
        self.register_buffer("p_mean", torch.as_tensor(prop_mean))
        self.register_buffer("p_std", torch.as_tensor(prop_std))
        self.cnn = make_cnn()          # camera fixe 'front'
        self.cnn_wrist = make_cnn()    # camera poignet 'wrist'
        self.head = nn.Sequential(
            nn.Linear(256 + 256 + 8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4), nn.Tanh(),
        )

    @staticmethod
    def to_float(img_uint8):
        return img_uint8.permute(0, 3, 1, 2).float() / 255.0     # (B, 96, 96, 3) -> (B, 3, 96, 96)

    def forward(self, img, wrist, prop):
        f_front = self.cnn(self.to_float(img))
        f_wrist = self.cnn_wrist(self.to_float(wrist))
        p = (prop - self.p_mean) / self.p_std
        return self.head(torch.cat([f_front, f_wrist, p], dim=1))


def act_from_env(policy, env):
    """Lit les 2 cameras + la proprio dans l'env, renvoie (action, image fixe, image poignet)."""
    front, wrist = env.camera_image(), env.wrist_image()
    with torch.no_grad():
        act = policy(torch.as_tensor(front).unsqueeze(0), torch.as_tensor(wrist).unsqueeze(0),
                     torch.as_tensor(env.proprio()).unsqueeze(0))[0].numpy()
    return act, front, wrist


def random_shift(img):
    """Decale chaque image de quelques pixels (bords repetes) : le modele ne doit pas
    dependre d'un pixel exact, ce qui aide beaucoup a generaliser."""
    b, h, w, c = img.shape
    x = F.pad(img.permute(0, 3, 1, 2).float(), (SHIFT,) * 4, mode="replicate")
    out = torch.empty((b, c, h, w))
    dx = torch.randint(0, 2 * SHIFT + 1, (b,))
    dy = torch.randint(0, 2 * SHIFT + 1, (b,))
    for i in range(b):
        out[i] = x[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w]
    return out.permute(0, 2, 3, 1)


def batched_loss(policy, I, W, P, A, size=512):
    """Loss moyenne sur un gros jeu de donnees, par paquets (sans gradient)."""
    with torch.no_grad():
        return sum(((policy(I[j:j + size], W[j:j + size], P[j:j + size]) - A[j:j + size]) ** 2).sum().item()
                   for j in range(0, len(I), size)) / (len(I) * 4)


def train():
    data = np.load("demos_vision.npz")
    if "wrist" not in data:
        raise SystemExit("demos_vision.npz n'a pas d'images poignet : relance python record_demos_vision.py")
    img, wrist, prop, act, ep = data["img"], data["wrist"], data["proprio"], data["act"], data["episode"]
    n_ep = ep.max() + 1
    print(f"{n_ep} episodes, {len(img)} pas, 2 images {img.shape[1]}x{img.shape[2]} par pas")

    val = np.isin(ep, np.arange(max(1, n_ep // 10)))               # 10 % des episodes en validation
    p_mean, p_std = prop[~val].mean(0), prop[~val].std(0) + 1e-6
    T = lambda a: torch.as_tensor(a)                                # noqa: E731
    I, W, P, A = T(img[~val]), T(wrist[~val]), T(prop[~val]), T(act[~val])
    Iv, Wv, Pv, Av = T(img[val]), T(wrist[val]), T(prop[val]), T(act[val])

    policy = VisionPolicy(p_mean, p_std)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    # le pas d'apprentissage descend doucement jusqu'a ~0 en fin d'entrainement (affine la precision)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        policy.train()
        perm = torch.randperm(len(I))
        total = 0.0
        for i in range(0, len(I), BATCH):
            idx = perm[i:i + BATCH]
            pred = policy(random_shift(I[idx]), random_shift(W[idx]), P[idx])
            loss = ((pred - A[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        policy.eval()
        val_loss = batched_loss(policy, Iv, Wv, Pv, Av)
        print(f"epoch {epoch:3d}  loss entrainement {total / len(I):.4f}  "
              f"loss validation {val_loss:.4f}  ({time.time() - t0:.0f} s)")

    torch.save(policy.state_dict(), "bc_vision.pt")
    print("Modele sauve dans bc_vision.pt")
    return policy


def model_path():
    """Chemin du modele : --model fichier.pt, sinon bc_vision.pt."""
    if "--model" in sys.argv:
        return sys.argv[sys.argv.index("--model") + 1]
    return "bc_vision.pt"


def load_policy(path=None):
    sd = torch.load(path or model_path())
    if "cnn_wrist.0.weight" not in sd:
        raise SystemExit("Ce modele date d'avant la camera poignet : relance python train_bc_vision.py")
    policy = VisionPolicy(sd["p_mean"].numpy(), sd["p_std"].numpy())
    policy.load_state_dict(sd)
    return policy


def show_cameras(front, wrist):
    """Affiche les 2 images exactes donnees au modele, cote a cote, agrandies 4x."""
    import cv2
    both = np.concatenate([front, wrist], axis=1)[:, :, ::-1]
    big = cv2.resize(both, (IMG_SIZE * 8, IMG_SIZE * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Ce que voit le modele : camera fixe | camera poignet", big)
    cv2.waitKey(1)


def evaluate(policy, view=False):
    n_ep = 5 if view else 50
    env = PandaPickEnv(render_mode="human" if view else None)
    cam_window = False
    if view:
        try:
            import cv2  # noqa: F401
            cam_window = True
        except ImportError:
            print("(fenetre camera desactivee : python -m pip install opencv-python)")
    policy.eval()
    successes = 0
    for e in range(n_ep):
        env.reset(seed=e)                      # seeds 0..49 : jamais vues a l'entrainement
        done, success = False, False
        while not done:
            act, front, wrist = act_from_env(policy, env)
            if cam_window:
                show_cameras(front, wrist)
            _, _, term, trunc, info = env.step(act)     # l'observation "etat" est ignoree
            success = info["is_success"]                # juge a la FIN
            done = term or trunc
            if view:
                time.sleep(0.04)
        successes += success
        print(f"episode {e:2d} : {'REUSSI' if success else 'echec'}")
    print(f"\nTaux de reussite du modele BC vision : {successes}/{n_ep}")
    env.close()
    if cam_window:
        import cv2
        cv2.destroyAllWindows()


if __name__ == "__main__":
    policy = load_policy() if "--eval" in sys.argv else train()
    evaluate(policy, view="--view" in sys.argv)
