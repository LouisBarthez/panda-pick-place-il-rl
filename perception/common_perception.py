"""Briques partagees par les scripts de perception/ : chemins, creation de l'env avec la camera
RGB-D, boucle d'un episode avec l'etat exact ou l'etat estime par la camera."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for sub in ("common", "perception"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from detector import ObjectTracker, estimated_obs  # noqa: E402
from panda_pick_env import CUBE_HALF, MAX_STEPS, PandaPickEnv  # noqa: E402
from realsense import RealSense, add_realsense  # noqa: E402

N_TEST = 50          # memes episodes de test que le README : seeds 0..49


def make_env(shadows=False, noise=True, calib_error=(0.0, 0.0)):
    env = PandaPickEnv(add_to_spec=add_realsense)
    cam = RealSense(env, shadows=shadows, noise=noise, calib_error=calib_error)
    return env, cam, ObjectTracker(cam)


def run_episode(env, tracker, policy, seed, use_estimate, on_step=None):
    """Joue un episode. policy(obs, state) -> action.
    use_estimate=False : la politique recoit l'etat exact (comme dans le README), la camera
                         tourne quand meme pour mesurer l'erreur de perception ;
    use_estimate=True  : la politique recoit cube et cible ESTIMES par la camera.
    Renvoie (reussite, journal) ; le journal a une ligne par pas :
    erreur cube [m], erreur cible [m], cube detecte, cible detectee, cube en l'air."""
    obs, _ = env.reset(seed=seed)
    tracker.reset()
    tracker.cam.reseed(1000 + seed)
    state, log, info = {}, [], {"is_success": False}
    for t in range(MAX_STEPS):
        cube_est, target_est = tracker.update()
        cube_true = env.data.xpos[env.cube_body].copy()
        if cube_est is None or target_est is None:
            # rien vu sur la toute premiere image : on n'a aucune estimation a donner
            cube_err = target_err = np.nan
            obs_policy = obs if not use_estimate else estimated_obs(
                obs, cube_est if cube_est is not None else obs[0:3],
                target_est if target_est is not None else obs[0:3])
        else:
            cube_err = np.linalg.norm(cube_est - cube_true)
            target_err = np.linalg.norm(target_est - env.target_pos)
            obs_policy = estimated_obs(obs, cube_est, target_est) if use_estimate else obs
        log.append((cube_err, target_err, tracker.last["cube"] is not None,
                    tracker.last["target"] is not None, cube_true[2] > CUBE_HALF + 0.005))
        action = policy(obs_policy, state)
        if on_step is not None:
            on_step(env, tracker, t, obs_policy, cube_true)
        obs, _, _, _, info = env.step(action)
    return bool(info["is_success"]), np.array(log, dtype=float)
