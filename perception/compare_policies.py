"""Etat exact vs etat estime par la camera : meme politique, memes 50 episodes de test.

Pour chaque politique, on joue les seeds 0..49 deux fois :
  - "exact"  : cube et cible lus dans le simulateur (comme dans le README)
  - "estime" : cube et cible estimes par la camera RGB-D (perception/detector.py)
Le reste de l'observation (outil, vitesse, pince) vient du robot dans les deux cas.
Les politiques ne sont PAS reentrainees : on teste si ce qui a ete appris (ou scripte) sur
l'etat exact supporte l'erreur d'une vraie chaine de perception.

    python perception/compare_policies.py              -> expert + SAC (si rl/sac_best.zip existe)
    python perception/compare_policies.py 10           -> 10 episodes
    python perception/compare_policies.py --expert     -> expert seulement
    python perception/compare_policies.py --sac        -> SAC seulement
Sorties : perception/results/compare_policies.json (+ resume a l'ecran). Les resultats d'une
politique non rejouee sont gardes s'ils existent deja dans le fichier.
"""
import json
import os
import sys
import time

from common_perception import N_TEST, ROOT, HERE, make_env, run_episode

import numpy as np  # noqa: E402

from test_expert import expert_action  # noqa: E402

N_EP = next((int(a) for a in sys.argv[1:] if a.isdigit()), N_TEST)
SHADOWS = "--shadows" in sys.argv


def load_sac():
    rl = os.path.join(ROOT, "rl")
    if not os.path.exists(os.path.join(rl, "sac_best.zip")):
        return None
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from panda_pick_env import PandaPickEnv
    venv = VecNormalize.load(os.path.join(rl, "vecnormalize_best.pkl"), DummyVecEnv([PandaPickEnv]))
    venv.training = False
    model = SAC.load(os.path.join(rl, "sac_best"), device="cpu")

    def policy(obs, state):
        act, _ = model.predict(venv.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
        return act[0]
    return policy


def main():
    policies = {} if "--sac" in sys.argv else {"expert": expert_action}
    if "--expert" not in sys.argv:
        sac = load_sac()
        if sac is None:
            print("rl/sac_best.zip absent : seul l'expert est compare (entrainer avec rl/train_rl.py)")
        else:
            policies["sac"] = sac

    env, cam, tracker = make_env(shadows=SHADOWS)
    path = os.path.join(HERE, "results", "compare_policies.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        if results.get("episodes") != N_EP:
            results = {}
    results.update({"episodes": N_EP, "seeds": f"0..{N_EP - 1}"})
    t0 = time.time()
    for name, policy in policies.items():
        for mode in ("exact", "estime"):
            succ, errs = [], []
            for ep in range(N_EP):
                ok, log = run_episode(env, tracker, policy, ep, use_estimate=(mode == "estime"))
                succ.append(ok)
                errs.append(log[:, 0])
            errs = np.concatenate(errs)
            results[f"{name}_{mode}"] = {
                "reussite": f"{sum(succ)}/{N_EP}",
                "echecs_seeds": [i for i, s in enumerate(succ) if not s],
                "erreur_cube_moyenne_cm": round(float(np.nanmean(errs)) * 100, 2),
                "erreur_cube_max_cm": round(float(np.nanmax(errs)) * 100, 2),
            }
            print(f"{name:6s} etat {mode:6s} : {sum(succ):2d}/{N_EP}   "
                  f"(erreur cube moyenne {np.nanmean(errs) * 100:.2f} cm)   {time.time() - t0:.0f} s", flush=True)
    env.close()
    cam.close()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
