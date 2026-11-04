"""Expert scripte pour PandaPickEnv : valide l'environnement, genere les demos
de l'imitation learning et donne les bonnes actions a DAgger.

    python test_expert.py              -> 50 episodes sans affichage, taux de reussite
    python test_expert.py --view       -> 5 episodes avec le viewer MuJoCo
"""
import sys
import time

import numpy as np

from panda_pick_env import CUBE_HALF, STEP_POS, PandaPickEnv

GAIN = 0.3   # gain du correcteur proportionnel (1.0 = toute l'erreur corrigee en un pas -> depassement)


def expert_action(obs, state):
    """Expert SANS MEMOIRE : il decide uniquement d'apres la scene a l'instant present.
    C'est indispensable pour DAgger : il doit pouvoir donner la bonne action meme dans une
    situation bizarre ou le modele s'est mis (cube rate, pince fermee a vide, etc.).
    state["phase"] sert seulement a afficher ce qu'il pense faire."""
    tcp, width, cube, target = obs[0:3], obs[6], obs[7:10], obs[13:16]

    def move_to(goal, grip):
        delta = GAIN * (goal - tcp) / STEP_POS
        return np.concatenate([np.clip(delta, -1, 1), [grip]]).astype(np.float32)

    d_xy = np.linalg.norm(cube[:2] - tcp[:2])           # ecart horizontal outil-cube
    dz = tcp[2] - cube[2]                                # hauteur de l'outil au-dessus du cube
    to_target = np.linalg.norm(target[:2] - cube[:2])    # ecart horizontal cube-cible
    in_hand = d_xy < 0.015 and abs(dz) < 0.015 and 0.03 < width < 0.045
    on_target = to_target < 0.012 and cube[2] < CUBE_HALF + 0.005

    if on_target and not (in_hand and dz > -0.01 and cube[2] > CUBE_HALF + 0.002):
        state["phase"] = "release/retreat"               # cube pose sur la cible : on lache et on remonte
        return move_to(target + [0, 0, 0.10], 1)
    if in_hand:
        if to_target > 0.015:
            if cube[2] < CUBE_HALF + 0.03:
                state["phase"] = "lift"                  # d'abord monter tout droit (ne pas trainer le cube)
                return move_to(np.array([cube[0], cube[1], CUBE_HALF + 0.08]), -1)
            state["phase"] = "transport"                 # puis aller au-dessus de la cible
            return move_to(np.array([target[0], target[1], CUBE_HALF + 0.08]), -1)
        state["phase"] = "place"                         # au-dessus de la cible : descendre
        return move_to(target, -1)
    if width < 0.03:
        state["phase"] = "reopen"                        # pince fermee a vide : on rouvre et on remonte
        return move_to(cube + [0, 0, 0.10], 1)
    if d_xy > 0.01:
        state["phase"] = "above"                         # se placer au-dessus du cube
        return move_to(cube + [0, 0, 0.10], 1)
    if dz > 0.008:
        state["phase"] = "descend"                       # aligne : descendre pince ouverte
        return move_to(cube, 1)
    state["phase"] = "close"                             # a hauteur du cube : fermer
    return move_to(cube, -1)


def main():
    view = "--view" in sys.argv
    n_ep = 5 if view else 50
    env = PandaPickEnv(render_mode="human" if view else None)
    successes = 0
    for ep in range(n_ep):
        obs, _ = env.reset(seed=ep)
        state = {"phase": "above"}
        done, success = False, False
        while not done:
            obs, r, term, trunc, info = env.step(expert_action(obs, state))
            success = info["is_success"]   # juge a la FIN : le cube doit etre reste sur la cible
            done = term or trunc
            if view:
                time.sleep(0.04)
        successes += success
        print(f"episode {ep:2d} : {'REUSSI' if success else 'echec'}  (phase finale : {state['phase']})")
    print(f"\nTaux de reussite de l'expert : {successes}/{n_ep}")
    env.close()


if __name__ == "__main__":
    main()
