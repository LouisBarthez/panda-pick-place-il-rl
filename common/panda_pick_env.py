"""Environnement Gymnasium : Panda (mujoco_menagerie) + cube, tache pick-and-place.

Action (4 valeurs dans [-1, 1]) :
    dx, dy, dz  -> deplacement de la cible de l'outil (max STEP_POS m par pas)
    pince       -> -1 = fermee, +1 = ouverte
L'orientation de la pince reste verticale (vers le bas). Un IK differentiel convertit
la cible cartesienne en consignes articulaires pour les actionneurs en position du modele.

Observation "etat" (19 valeurs, utilisee par l'expert et le BC etat) :
    pos outil (3), vitesse outil (3), ouverture pince (1),
    pos cube (3), cube - outil (3), pos cible (3), cible - cube (3)
Observation "vision" (BC vision) : env.camera_image() + env.wrist_image() + env.proprio()
    image 96x96 de la camera fixe 'front' + image 96x96 de la camera poignet 'wrist'
    + 7 angles articulaires + ouverture pince.
La cible est un marquage vert au sol, visible sur l'image.

Utilisation :
    from panda_pick_env import PandaPickEnv
    env = PandaPickEnv(render_mode="human")
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

PANDA_DIR = os.environ.get(
    "PANDA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mujoco_menagerie", "franka_emika_panda"),
)

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034           # bride de la main -> centre des doigts [m]
CUBE_HALF = 0.02              # cube de 4 cm
FRAME_SKIP = 20               # 20 x 2 ms = 40 ms par pas (25 Hz)
STEP_POS = 0.02               # deplacement max de la cible par pas [m]
MAX_STEPS = 100               # 4 s par episode
SUCCESS_DIST = 0.03           # reussite = cube a moins de 3 cm de la cible ET lache (pince ouverte > 4,5 cm)

# zone de travail (repere base du robot)
CUBE_X, CUBE_Y = (0.45, 0.65), (-0.15, 0.15)
IMG_SIZE = 96                 # image camera (carree, en pixels)
WS_LOW = np.array([0.30, -0.30, 0.0])
WS_HIGH = np.array([0.75, 0.30, 0.40])


def build_model():
    spec = mujoco.MjSpec.from_file(os.path.join(PANDA_DIR, "scene.xml"))
    # la keyframe "home" ne connait pas le cube : on la supprime et on regle qpos a la main
    for key in list(spec.keys):
        spec.delete(key)
    for mat in spec.materials:          # sol sans reflet : image plus simple a lire pour le CNN
        mat.reflectance = 0.0

    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0],
                               rgba=[1, 0, 0, 0])   # invisible : il n'existe pas sur le vrai robot
    # camera poignet : fixee sur la main, legerement en retrait, elle regarde entre les doigts
    spec.body("hand").add_camera(name="wrist", pos=[0.07, 0, 0.03],
                                 xyaxes=[0, 1, 0, 0.819, 0, 0.574], fovy=75)

    cube = spec.worldbody.add_body(name="cube", pos=[0.55, 0, CUBE_HALF])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=0.05, friction=[1.5, 0.01, 0.001], rgba=[0.85, 0.2, 0.2, 1])

    # cible = marquage vert au sol (pas de collision), comme un repere colle sur l'etabli
    target = spec.worldbody.add_body(name="target", mocap=True, pos=[0.55, 0.1, 0.0])
    target.add_geom(name="target_mark", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.001, 0],
                    pos=[0, 0, 0.001], rgba=[0.1, 0.8, 0.2, 1], contype=0, conaffinity=0)

    # camera fixe, devant le robot, qui regarde la zone de travail
    spec.worldbody.add_camera(name="front", pos=[0.95, 0, 0.65],
                              xyaxes=[0, 1, 0, -0.845, 0, 0.536], fovy=40)
    return spec.compile()


class PandaPickEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)   # brouillon pour l'IK
        self.render_mode = render_mode
        self.viewer = None
        self.cam_renderer = None                   # cree au premier appel de camera_image()

        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.f1 = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.f2 = m.jnt_qposadr[m.joint("finger_joint2").id]
        self.cube_qadr = m.jnt_qposadr[m.joint("cube_joint").id]
        self.tcp = m.site("tcp").id
        self.cube_body = m.body("cube").id
        self.target_mocap = m.body_mocapid[m.body("target").id]
        self.q_lo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.q_hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float64)

        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.R_down = None
        self.ee_target = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.steps = 0

    # ------------------------------------------------------------------ IK
    def _ik(self, pos_goal, iters=15):
        """IK differentiel (moindres carres amortis) : position + orientation verticale,
        avec un petit rappel vers la posture home dans l'espace nul."""
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p = d.site_xpos[self.tcp]
            R = d.site_xmat[self.tcp].reshape(3, 3)
            e_pos = pos_goal - p
            # erreur d'orientation (repere monde)
            e_rot = 0.5 * (np.cross(R[:, 0], self.R_down[:, 0]) + np.cross(R[:, 1], self.R_down[:, 1])
                           + np.cross(R[:, 2], self.R_down[:, 2]))
            mujoco.mj_jacSite(self.model, d, self.jacp, self.jacr, self.tcp)
            J = np.vstack([self.jacp[:, self.arm_dof], self.jacr[:, self.arm_dof]])
            err = np.concatenate([e_pos, e_rot])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(JJt)
            q = d.qpos[self.arm_qadr]
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (0.1 * (HOME_Q - q))
            d.qpos[self.arm_qadr] = np.clip(q + dq, self.q_lo, self.q_hi)
        return d.qpos[self.arm_qadr].copy()

    # ------------------------------------------------------------------ Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_Q
        self.data.qpos[self.f1] = self.data.qpos[self.f2] = 0.04
        self.data.ctrl[:7] = HOME_Q
        self.data.ctrl[7] = 255

        rng = self.np_random
        cube_xy = np.array([rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y)])
        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = [*cube_xy, CUBE_HALF]
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]
        # cible a au moins 10 cm du cube
        while True:
            t = np.array([rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y), CUBE_HALF])
            if np.linalg.norm(t[:2] - cube_xy) > 0.10:
                break
        self.target_pos = t
        self.data.mocap_pos[self.target_mocap] = [t[0], t[1], 0.0]

        mujoco.mj_forward(self.model, self.data)
        self.R_down = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.ee_target = self.data.site_xpos[self.tcp].copy()
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.ee_target = np.clip(self.ee_target + a[:3] * STEP_POS, WS_LOW, WS_HIGH)
        self.data.ctrl[:7] = self._ik(self.ee_target)
        self.data.ctrl[7] = 255.0 if a[3] > 0 else 0.0
        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)
        self.steps += 1

        obs = self._obs()
        tcp, cube = self.data.site_xpos[self.tcp], self.data.xpos[self.cube_body]
        d_goal = np.linalg.norm(self.target_pos - cube)
        width = self.data.qpos[self.f1] + self.data.qpos[self.f2]
        # reussite = cube pose sur la cible ET lache (pince ouverte). Sans la 2e condition, le RL
        # avait appris a garder le cube serre sur la cible (01/11) : il faut ecrire toute la tache.
        success = d_goal < SUCCESS_DIST and width > 0.045   # meme seuil que "tenu" : pas de creux
        reward = self._reward(tcp, cube, d_goal, success)

        truncated = self.steps >= MAX_STEPS
        if self.render_mode == "human":
            self.render()
        return obs, reward, False, truncated, {"is_success": bool(success)}

    def _reward(self, tcp, cube, d_goal, success):
        """Recompense etagee (utilisee seulement par le RL ; l'IL l'ignore), par pas :
          approche : jusqu'a 0.25 quand l'outil est sur le cube (deja un peu des 50 cm)
          prise    : +0.25 tant que le cube est entre les doigts
          transport: jusqu'a 1.0 en plus quand le cube tenu est sur la cible
          reussite : 3 quand le cube est pose sur la cible ET lache (remplace le reste)
        Ordre des gains : tenir le cube loin de la cible (~0.75) < le tenir sur la cible (1.5)
        < le lacher sur la cible (3). Chaque etape rapporte nettement plus que la precedente,
        sinon le robot s'arrete a l'etape qui rapporte deja (essais du 01/11)."""
        d_grip = np.linalg.norm(cube - tcp)
        width = self.data.qpos[self.f1] + self.data.qpos[self.f2]
        grasped = d_grip < 0.02 and 0.03 < width < 0.045
        # deux pentes : une douce (on sent le cube de loin), une raide (precision pres du cube)
        near = lambda d: 0.5 * (1 - np.tanh(3 * d)) + 0.5 * (1 - np.tanh(15 * d))  # noqa: E731
        far = lambda d: 0.5 * (1 - np.tanh(2 * d)) + 0.5 * (1 - np.tanh(10 * d))  # noqa: E731
        if success:
            return 3.0
        r = 0.25 * near(d_grip)
        if grasped:
            r += 0.25 + 1.0 * far(d_goal)      # pente plus douce : la cible attire de plus loin
        return float(r)

    def _obs(self):
        tcp = self.data.site_xpos[self.tcp].copy()
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp)
        tcp_vel = self.jacp @ self.data.qvel
        width = self.data.qpos[self.f1] + self.data.qpos[self.f2]
        cube = self.data.xpos[self.cube_body].copy()
        return np.concatenate([tcp, tcp_vel, [width], cube, cube - tcp,
                               self.target_pos, self.target_pos - cube])

    # ------------------------------------------------------------------ vision
    def camera_image(self, camera="front"):
        """Image RGB (IMG_SIZE x IMG_SIZE x 3, uint8) d'une camera : 'front' (fixe) ou 'wrist' (poignet)."""
        if self.cam_renderer is None:
            self.cam_renderer = mujoco.Renderer(self.model, IMG_SIZE, IMG_SIZE)
        self.cam_renderer.update_scene(self.data, camera=camera)
        return self.cam_renderer.render().copy()

    def wrist_image(self):
        """Image de la camera poignet (meme taille que la camera fixe)."""
        return self.camera_image("wrist")

    def proprio(self):
        """Ce que le robot sait de lui-meme : 7 angles articulaires + ouverture de la pince."""
        q = self.data.qpos[self.arm_qadr]
        width = self.data.qpos[self.f1] + self.data.qpos[self.f2]
        return np.concatenate([q, [width]]).astype(np.float32)

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.cam_renderer is not None:
            self.cam_renderer.close()
            self.cam_renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
