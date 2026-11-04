"""Jusqu'ou la chaine tient-elle ? L'expert joue avec l'etat ESTIME dans des conditions degradees.

Conditions testees (memes 50 episodes, seeds 0..49) :
  - reference          : camera bruitee, calibration parfaite, sans ombres
  - ombres             : rendu avec ombres (le seuillage HSV doit les supporter)
  - calib 5 mm / 1 deg : la pose de la camera utilisee pour reprojeter est fausse
  - calib 10 mm / 2 deg
Pour le SAC, voir compare_policies.py (l'etude est faite sur l'expert, plus rapide a jouer).

    python perception/robustness.py          -> 50 episodes par condition (~30 min sur CPU)
    python perception/robustness.py 10
Sortie : perception/results/robustness.json
"""
import json
import os
import sys
import time

from common_perception import N_TEST, HERE, make_env, run_episode

import numpy as np  # noqa: E402

from test_expert import expert_action  # noqa: E402

N_EP = next((int(a) for a in sys.argv[1:] if a.isdigit()), N_TEST)
CONDITIONS = [
    ("reference", dict()),
    ("shadows", dict(shadows=True)),
    ("calibration error 5 mm / 1 deg", dict(calib_error=(5.0, 1.0))),
    ("calibration error 10 mm / 2 deg", dict(calib_error=(10.0, 2.0))),
]


def main():
    out, t0 = {"episodes": N_EP, "politique": "expert scripte, etat estime"}, time.time()
    for name, kw in CONDITIONS:
        env, cam, tracker = make_env(**kw)
        succ, cube_err, tgt_err = [], [], []
        for ep in range(N_EP):
            ok, log = run_episode(env, tracker, expert_action, ep, use_estimate=True)
            succ.append(ok)
            cube_err.append(log[:, 0])
            tgt_err.append(log[:, 1])
        env.close()
        cam.close()
        c, g = np.concatenate(cube_err), np.concatenate(tgt_err)
        out[name] = {"reussite": f"{sum(succ)}/{N_EP}", "echecs_seeds": [i for i, s in enumerate(succ) if not s],
                     "erreur_cube_moyenne_cm": round(float(np.nanmean(c)) * 100, 2),
                     "erreur_cube_max_cm": round(float(np.nanmax(c)) * 100, 2),
                     "erreur_cible_moyenne_cm": round(float(np.nanmean(g)) * 100, 2),
                     "erreur_cible_max_cm": round(float(np.nanmax(g)) * 100, 2)}
        print(f"{name:34s} {sum(succ):2d}/{N_EP}   cube {np.nanmean(c) * 100:.2f} cm (max {np.nanmax(c) * 100:.2f})"
              f"   cible {np.nanmean(g) * 100:.2f} cm   ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "robustness.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
