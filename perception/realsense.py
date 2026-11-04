"""Camera RGB-D simulee, inspiree d'une Intel RealSense D435 (couleur + profondeur alignee).

Ce qui est imite d'une vraie camera :
  - resolution 424 x 240 (un mode reel de la D435) et champ vertical de 42 deg (capteur couleur),
  - profondeur alignee sur l'image couleur, en metres, quantifiee au millimetre,
  - bruit de profondeur qui croit avec le carre de la distance, et pixels sans mesure (trous),
  - bruit sur l'image couleur.
Ce qui reste idealise (a garder en tete pour le vrai robot) :
  - intrinseques et position de la camera (extrinseques) parfaitement connues par defaut : en
    vrai, elles viennent d'une calibration, qui a sa propre erreur (option calib_error pour
    en simuler une) ;
  - pas de flou de mouvement, pas de reflets, eclairage stable, image et profondeur synchrones.

Utilisation :
    env = PandaPickEnv(add_to_spec=add_realsense)
    cam = RealSense(env)
    rgb, depth = cam.capture()           # (240, 424, 3) uint8, (240, 424) float32 en m (0 = pas de mesure)
    p_world = cam.pixel_to_world(u, v, z)
"""
import mujoco
import numpy as np

NAME = "realsense"
WIDTH, HEIGHT = 424, 240
FOVY_DEG = 42.0                     # champ vertical du capteur couleur de la D435
# Camera sur un trepied devant la table, en hauteur, qui plonge vers la zone de travail.
# Placee face au robot : les doigts de la pince s'ouvrent selon y, la camera voit donc le cube
# ENTRE les doigts pendant la prise (vue de cote, un doigt le cacherait).
POS = np.array([0.60, -0.75, 0.55])
LOOK_AT = np.array([0.55, 0.0, 0.02])

DEPTH_MIN, DEPTH_MAX = 0.2, 3.0     # portee utile [m]
DEPTH_NOISE = 0.002                 # ecart-type du bruit de profondeur a 1 m [m] (croit en z^2)
DEPTH_HOLES = 0.01                  # proportion de pixels sans mesure
RGB_NOISE = 3.0                     # ecart-type du bruit couleur (niveaux sur 255)


def _xyaxes(pos, look_at):
    """Axes x (droite) et y (haut) d'une camera MuJoCo qui regarde look_at depuis pos.
    Une camera MuJoCo regarde selon son -z."""
    f = look_at - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y])


def add_realsense(spec):
    """A passer a PandaPickEnv(add_to_spec=...) : ajoute la camera (ne change pas la physique)."""
    spec.worldbody.add_camera(name=NAME, pos=POS.tolist(), xyaxes=_xyaxes(POS, LOOK_AT).tolist(),
                              fovy=FOVY_DEG)


class RealSense:
    def __init__(self, env, seed=0, shadows=False, noise=True, calib_error=(0.0, 0.0)):
        """shadows=False : rendu sans ombres, beaucoup plus rapide en rendu logiciel (CPU sans GPU).
        noise=False : image et profondeur parfaites (pour isoler l'erreur due a la methode).
        calib_error=(mm, deg) : la pose de la camera CROYANCE differe de la vraie d'une translation
        de `mm` millimetres et d'une rotation de `deg` degres (directions fixes) : simule une
        calibration main-oeil imparfaite. (0, 0) = calibration parfaite."""
        self.env = env
        self.model, self.data = env.model, env.data
        self.cam_id = self.model.camera(NAME).id
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.shadows = shadows
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # intrinseques (modele stenope, pixels carres, point principal au centre de l'image)
        self.fy = 0.5 * HEIGHT / np.tan(np.radians(FOVY_DEG) / 2)
        self.fx = self.fy
        self.cx, self.cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
        # erreur de calibration : directions tirees une fois pour toutes (graine fixe)
        g = np.random.default_rng(12345)
        dp, axis = g.normal(size=3), g.normal(size=3)
        self.calib_dp = calib_error[0] / 1000 * dp / np.linalg.norm(dp)
        axis = axis / np.linalg.norm(axis)
        a = np.radians(calib_error[1])
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        self.calib_dR = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx   # Rodrigues

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def K(self):
        """Matrice des intrinseques (comme rs.intrinsics sur une vraie RealSense)."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def _render(self, depth):
        if depth:
            self.renderer.enable_depth_rendering()
        else:
            self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        if not self.shadows:
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return self.renderer.render().copy()

    def capture(self):
        rgb = self._render(depth=False)
        depth = self._render(depth=True).astype(np.float32)   # profondeur le long de l'axe optique [m]
        if self.noise:
            rgb = np.clip(rgb + self.rng.normal(0, RGB_NOISE, rgb.shape), 0, 255).astype(np.uint8)
            depth = depth + self.rng.normal(0, 1, depth.shape) * DEPTH_NOISE * depth ** 2
            depth = np.round(depth * 1000) / 1000                       # unite de la D435 : 1 mm
            depth[self.rng.random(depth.shape) < DEPTH_HOLES] = 0.0     # trous
        depth[(depth < DEPTH_MIN) | (depth > DEPTH_MAX)] = 0.0
        return rgb, depth.astype(np.float32)

    def true_pose(self):
        return self.data.cam_xpos[self.cam_id].copy(), self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def pose(self):
        """Pose de la camera telle que la perception la CROIT (en vrai : resultat de la calibration
        main-oeil). Egale a la vraie pose si calib_error = (0, 0)."""
        pos, R = self.true_pose()
        return pos + self.calib_dp, self.calib_dR @ R

    def pixels_to_world(self, u, v, z):
        """Reprojection : pixels (u colonne, v ligne) + profondeur z [m] -> points 3D dans le monde.
        Accepte des tableaux. Repere camera MuJoCo : x a droite, y en haut, regard selon -z."""
        u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
        p_cam = np.stack([(u - self.cx) / self.fx * z, -(v - self.cy) / self.fy * z, -z], axis=-1)
        pos, R = self.pose()
        return p_cam @ R.T + pos

    def world_to_pixel(self, p):
        """Projection avec la vraie pose (pour dessiner des positions sur l'image)."""
        pos, R = self.true_pose()
        pc = (np.asarray(p) - pos) @ R
        z = -pc[..., 2]
        return self.cx + self.fx * pc[..., 0] / z, self.cy - self.fy * pc[..., 1] / z

    def close(self):
        self.renderer.close()
