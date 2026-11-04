"""GIF + MP4 de la perception en boucle fermee : la politique ne recoit QUE l'etat estime.

A gauche : la vue de la camera RGB-D avec la detection dessinee
           (contour + centre estime, croix blanche = position vraie, erreur en mm).
           L'image affichee est celle de la camera sans bruit (GIF plus leger) ; la detection
           dessinee a ete calculee sur l'image bruitee, comme dans les evaluations.
A droite : la scene.

    python perception/make_gif.py               -> SAC si rl/sac_best.zip existe, sinon l'expert
    python perception/make_gif.py --expert      -> l'expert scripte
    python perception/make_gif.py --episodes 3
Sorties : media/perception.gif, media/perception.mp4
"""
import os
import sys

from common_perception import ROOT, make_env, run_episode

import cv2  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from realsense import RealSense  # noqa: E402
from test_expert import expert_action  # noqa: E402

N_EP = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 2
SEEDS = [3, 7, 11, 12, 15][:N_EP]
SCENE_W, SCENE_H = 400, 300
CAM_SCALE = 300 / 240              # vue camera mise a la meme hauteur que la scene
EVERY = 2                          # un pas sur 2 dans le GIF : 10 images/s, a peu pres le temps reel
HOLD = 8


class SceneCamera:
    """Meme vue que videos/make_videos.py (rendu sans ombres : plus propre en rendu logiciel)."""

    def __init__(self, env):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, SCENE_H, SCENE_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.42, 0.0, 0.3]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.25, 140, -20

    def frame(self):
        self.renderer.update_scene(self.env.data, camera=self.cam)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        return self.renderer.render().copy()


def text(img, lines, y0=18, color=(255, 255, 255), scale=0.45):
    for i, t in enumerate(lines):
        y = y0 + 18 * i
        (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (4, y - h - 4), (10 + w, y + 4), (0, 0, 0), -1)
        cv2.putText(img, t, (7, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def draw_detection(tracker, display_cam, cube_true, target_true):
    last, cam = tracker.last, tracker.cam
    # Affichage seulement : on montre l'image SANS bruit de la meme camera au meme instant (le bruit
    # change a chaque image et rendrait le GIF 5 fois plus lourd). La detection dessinee, elle,
    # a ete faite sur l'image bruitee.
    clean, _ = display_cam.capture()
    img = cv2.resize(clean, None, fx=CAM_SCALE, fy=CAM_SCALE, interpolation=cv2.INTER_LINEAR)
    s = CAM_SCALE
    lines = []
    for key, est, true, col, name in (("cube", tracker.cube, cube_true, (255, 230, 0), "cube"),
                                      ("target", tracker.target, target_true, (0, 230, 255), "target")):
        det = last[key]
        if det is not None:
            cv2.drawContours(img, [np.round(det["contour"] * s).astype(np.int32)], -1, col, 2, cv2.LINE_AA)
        if est is not None:
            u, v = cam.world_to_pixel(est)
            cv2.circle(img, (int(u * s), int(v * s)), 3, col, -1, cv2.LINE_AA)
            ut, vt = cam.world_to_pixel(true)
            cv2.drawMarker(img, (int(ut * s), int(vt * s)), (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
            err = np.linalg.norm(est - true) * 1000
            lines.append(f"{name}: error {err:4.1f} mm" + ("" if det is not None else " (not seen, last estimate)"))
    return text(img, ["simulated RGB-D camera (424x240)"] + lines)


def save_gif(path, frames, ms=100):
    """GIF leger : une seule palette de 256 couleurs pour toutes les images, sans tramage
    (le tramage change d'une image a l'autre et empeche la compression)."""
    from PIL import Image
    sample = np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0)
    # nuancier ajoute a l'echantillon : sans lui, les petites taches rouge et verte (cube, cible)
    # n'ont pas de couleur dans la palette et sortent marron / delavees
    ramp = np.linspace(0.35, 1.0, 16)[:, None]
    keys = np.array([[217, 51, 51], [26, 204, 51], [255, 230, 0], [0, 230, 255], [120, 255, 120],
                     [255, 120, 120]], float)
    swatch = np.concatenate([np.repeat(np.repeat((ramp * k)[:, None, :], 40, 1), 40, 0) for k in keys], 1)
    swatch = cv2.resize(swatch.astype(np.uint8), (sample.shape[1], 16 * 40))
    sample = np.concatenate([sample, swatch], axis=0)
    palette = Image.fromarray(sample).quantize(256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def main():
    use_sac = "--expert" not in sys.argv and os.path.exists(os.path.join(ROOT, "rl", "sac_best.zip"))
    if use_sac:
        from compare_policies import load_sac
        policy, title = load_sac(), "SAC, estimated state"
    else:
        policy, title = expert_action, "scripted expert, estimated state"
    env, cam, tracker = make_env()
    display_cam = RealSense(env, noise=False)
    scene = SceneCamera(env)
    frames = []

    def on_step(env, tracker, t, obs, cube_true):
        if t % EVERY:
            return
        left = draw_detection(tracker, display_cam, cube_true, env.target_pos)
        right = text(scene.frame(), [title, f"step {t + 1}/100"])
        frames.append(np.concatenate([left, right], axis=1))

    for seed in SEEDS:
        ok, _ = run_episode(env, tracker, policy, seed, use_estimate=True, on_step=on_step)
        last = frames[-1].copy()
        text(last[:, -SCENE_W:], ["SUCCESS" if ok else "FAILURE"], y0=60,
             color=(120, 255, 120) if ok else (255, 120, 120), scale=0.7)
        frames += [last] * HOLD
        print(f"seed {seed} : {'reussi' if ok else 'echec'}")
    env.close()
    media = os.path.join(ROOT, "media")
    save_gif(os.path.join(media, "perception.gif"), frames)
    imageio.mimsave(os.path.join(media, "perception.mp4"), frames, fps=10, codec="libx264",
                    quality=8, macro_block_size=2)
    print("GIF :", os.path.normpath(os.path.join(media, "perception.gif")))


if __name__ == "__main__":
    main()
