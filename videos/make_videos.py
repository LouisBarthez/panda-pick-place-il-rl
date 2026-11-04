"""Enregistre des videos MP4 de la tache faite par l'expert, l'IL (BC vision) et le RL (SAC).

    python videos\\make_videos.py            -> 3 episodes par methode
    python videos\\make_videos.py 5          -> 5 episodes par methode

Sorties dans le dossier videos/ :
    expert.mp4             vue de la scene
    il_bc_vision.mp4       vue de la scene + les 2 cameras du modele (fixe en haut, poignet en bas)
    rl_sac.mp4             vue de la scene
Installation (une fois) : python -m pip install imageio imageio-ffmpeg opencv-python
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "il"))

import cv2  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from panda_pick_env import PandaPickEnv  # noqa: E402
from test_expert import expert_action  # noqa: E402

N_EP = next((int(a) for a in sys.argv[1:] if a.isdigit()), 3)
W, H = 640, 480          # taille de la vue de la scene
FPS = 25                 # 1 pas de simulation = 40 ms : la video est en temps reel
PANEL = H // 2           # taille d'une vignette camera (240 x 240)
HOLD = 12                # images figees en fin d'episode (environ 0,5 s) pour voir le resultat


class SceneCamera:
    """Camera libre qui filme toute la zone de travail (n'est vue par aucun modele)."""

    def __init__(self, env):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.42, 0.0, 0.3]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.25, 140, -20

    def frame(self):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        return self.renderer.render().copy()


def label(img, lines, color=(255, 255, 255)):
    """Ecrit du texte avec un fond sombre en haut a gauche de l'image."""
    img = img.copy()
    for i, txt in enumerate(lines):
        y = 26 + 26 * i
        cv2.rectangle(img, (8, y - 19), (14 + 11 * len(txt), y + 7), (0, 0, 0), -1)
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def with_cameras(scene, front, wrist):
    """Colle a droite de la scene les 2 images vues par le modele (agrandies, pixels visibles)."""
    f = cv2.resize(front, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    w = cv2.resize(wrist, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    f = label(f, ["camera fixe"])
    w = label(w, ["camera poignet"])
    return np.concatenate([scene, np.concatenate([f, w], axis=0)], axis=1)


def record(name, title, policy_fn, show_cams=False):
    env = PandaPickEnv()
    cam = SceneCamera(env)
    path = os.path.join(HERE, name)
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    ok = 0
    for e in range(N_EP):
        obs, _ = env.reset(seed=e)
        state = {}
        for t in range(100):
            act, front, wrist = policy_fn(env, obs, state)
            img = label(cam.frame(), [title, f"episode {e + 1}/{N_EP}   pas {t + 1}/100"])
            if show_cams:
                img = with_cameras(img, front, wrist)
            writer.append_data(img)
            obs, _, _, _, info = env.step(act)
        success = info["is_success"]
        ok += success
        img = label(cam.frame(), [title, f"episode {e + 1}/{N_EP}",
                                  "REUSSI" if success else "ECHEC"],
                    color=(120, 255, 120) if success else (255, 120, 120))
        if show_cams:
            img = with_cameras(img, env.camera_image(), env.wrist_image())
        for _ in range(HOLD):
            writer.append_data(img)
    writer.close()
    env.close()
    print(f"{name} : {ok}/{N_EP} reussis  ->  {path}")


def expert_fn(env, obs, state):
    return expert_action(obs, state), None, None


def make_il_fn():
    cwd = os.getcwd()
    from train_bc_vision import act_from_env, load_policy   # ce module change de dossier a l'import
    os.chdir(cwd)
    policy = load_policy(os.path.join(ROOT, "il", "bc_vision.pt"))
    policy.eval()
    return lambda env, obs, state: act_from_env(policy, env)


def make_rl_fn():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    rl = os.path.join(ROOT, "rl")
    venv = VecNormalize.load(os.path.join(rl, "vecnormalize_best.pkl"), DummyVecEnv([PandaPickEnv]))
    venv.training = False
    model = SAC.load(os.path.join(rl, "sac_best"))

    def fn(env, obs, state):
        act, _ = model.predict(venv.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
        return act[0], None, None
    return fn


def main():
    record("expert.mp4", "Expert scripte (etat exact)", expert_fn)
    if os.path.exists(os.path.join(ROOT, "il", "bc_vision.pt")):
        record("il_bc_vision.mp4", "IL : BC vision (2 cameras + angles)", make_il_fn(), show_cams=True)
    else:
        print("il/bc_vision.pt absent : video IL sautee")
    if os.path.exists(os.path.join(ROOT, "rl", "sac_best.zip")):
        record("rl_sac.mp4", "RL : SAC (etat exact)", make_rl_fn())
    else:
        print("rl/sac_best.zip absent : video RL sautee")


if __name__ == "__main__":
    main()
