"""Erreur de la perception : position estimee par la camera RGB-D vs position vraie (simulateur).

L'expert joue avec l'etat EXACT (memes trajectoires que dans le README) ; la camera tourne a
chaque pas et on compare ses estimations a la verite. On separe trois cas :
  - cube pose sur la table (la camera le voit en entier, sauf quand la pince arrive dessus)
  - cube dans la pince, en l'air (la main cache le dessus, les doigts une partie des cotes)
  - cible (marquage vert immobile, cachee en partie par le bras puis par le cube pose dessus)

    python perception/eval_perception.py              -> 50 episodes (seeds 0..49), ~5-10 min sur CPU
    python perception/eval_perception.py 10           -> 10 episodes
    python perception/eval_perception.py --no-noise   -> camera parfaite (isole l'erreur de la methode)
    python perception/eval_perception.py --shadows    -> rendu avec ombres (plus lent)
Sorties : perception/results/perception_errors.csv, perception_summary.json,
          media/perception_error.png
"""
import json
import os
import sys
import time

from common_perception import N_TEST, ROOT, HERE, make_env, run_episode

import numpy as np  # noqa: E402

from test_expert import expert_action  # noqa: E402

N_EP = next((int(a) for a in sys.argv[1:] if a.isdigit()), N_TEST)
NOISE = "--no-noise" not in sys.argv
SHADOWS = "--shadows" in sys.argv
OUT = os.path.join(HERE, "results")


def stats_cm(x):
    x = x[~np.isnan(x)] * 100
    return {"moyenne_cm": round(float(x.mean()), 2), "p95_cm": round(float(np.percentile(x, 95)), 2),
            "max_cm": round(float(x.max()), 2), "n_images": int(len(x))}


def main():
    os.makedirs(OUT, exist_ok=True)
    env, cam, tracker = make_env(shadows=SHADOWS, noise=NOISE)
    rows, t0, ok = [], time.time(), 0
    for ep in range(N_EP):
        success, log = run_episode(env, tracker, expert_action, ep, use_estimate=False)
        ok += success
        for t, r in enumerate(log):
            rows.append((ep, t, *r))
        print(f"episode {ep:2d}  cube {np.nanmean(log[:, 0]) * 100:4.2f} cm  "
              f"cible {np.nanmean(log[:, 1]) * 100:4.2f} cm  ({time.time() - t0:.0f} s)", flush=True)
    env.close()
    cam.close()
    a = np.array(rows, dtype=float)
    np.savetxt(os.path.join(OUT, "perception_errors.csv"), a, delimiter=",", comments="", fmt="%.6g",
               header="episode,pas,erreur_cube_m,erreur_cible_m,cube_detecte,cible_detectee,cube_en_l_air")

    air = a[:, 6] > 0
    summary = {
        "episodes": N_EP, "bruit_camera": NOISE, "ombres": SHADOWS,
        "reussite_expert_etat_exact": f"{ok}/{N_EP}",
        "cube_sur_la_table": stats_cm(a[~air, 2]),
        "cube_dans_la_pince": stats_cm(a[air, 2]),
        "cube_total": stats_cm(a[:, 2]),
        "cible": stats_cm(a[:, 3]),
        "cube_detecte_pct": round(100 * float(a[:, 4].mean()), 1),
        "cible_detectee_pct": round(100 * float(a[:, 5].mean()), 1),
    }
    tag = "" if NOISE else "_sans_bruit"
    with open(os.path.join(OUT, f"perception_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if NOISE:
        plot(a, air)


def plot(a, air):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = [a[~air, 2] * 100, a[air, 2] * 100, a[:, 3] * 100]
    data = [d[~np.isnan(d)] for d in data]
    labels = ["cube on table", "cube in gripper", "target"]
    ax[0].boxplot(data, whis=(0, 100))
    ax[0].set_xticks([1, 2, 3], labels)
    ax[0].set_ylabel("position error (cm)")
    ax[0].set_title("Estimated vs true position (whiskers = min/max)")
    ax[0].grid(alpha=0.3, axis="y")

    steps = int(a[:, 1].max()) + 1
    cube = np.array([np.nanmean(a[a[:, 1] == t, 2]) for t in range(steps)]) * 100
    cube_max = np.array([np.nanmax(a[a[:, 1] == t, 2]) for t in range(steps)]) * 100
    tgt = np.array([np.nanmean(a[a[:, 1] == t, 3]) for t in range(steps)]) * 100
    ax[1].plot(cube, label="cube (mean)")
    ax[1].plot(cube_max, ":", color="tab:blue", label="cube (max)")
    ax[1].plot(tgt, label="target (mean)")
    held = np.array([a[a[:, 1] == t, 6].mean() for t in range(steps)])
    ax[1].fill_between(range(steps), 0, 1, where=held > 0.5, transform=ax[1].get_xaxis_transform(),
                       color="grey", alpha=0.15, label="cube lifted (most episodes)")
    ax[1].set_xlabel("step (40 ms)")
    ax[1].set_ylabel("position error (cm)")
    ax[1].set_title(f"Over the episode ({len(np.unique(a[:, 0]))} episodes)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(ROOT, "media", "perception_error.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("Figure :", os.path.normpath(path))


if __name__ == "__main__":
    main()
