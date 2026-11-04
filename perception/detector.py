"""Detection du cube (rouge) et de la cible (marquage vert) par couleur, puis position 3D.

Pipeline classique, sans apprentissage :
  1. seuillage HSV (OpenCV) -> masque des pixels de la bonne couleur
  2. morphologie (ouverture puis fermeture) -> enleve les pixels isoles, bouche les petits trous
  3. plus grand contour -> l'objet ; les autres taches sont ignorees
  4. reprojection 3D : les pixels de l'objet qui ont une profondeur valide (sans le bord de 1 pixel)
     sont reprojetes dans le monde avec les intrinseques et la pose de la camera -> nuage de points
     de la surface visible ; medianes et percentiles le rendent robuste au bruit et aux trous
  5. correction de forme : la camera ne voit que la SURFACE de l'objet. Pour le cube (taille
     connue), le centre est un demi-cote sous la face du dessus et un demi-cote derriere la face
     de cote vue par la camera (voir cube_center). Pour la cible, la position utilisee par
     l'expert et le RL est le centre du cube pose dessus : on ajoute un demi-cote en z.
"""
import cv2
import numpy as np

CUBE_HALF = 0.02       # taille connue du cube (4 cm)
MARK_THICK = 0.002     # epaisseur connue du marquage de la cible
TOP_BAND = 0.004       # m : points a moins de 4 mm du plus haut = face du dessus du cube

# plages HSV d'OpenCV : H dans [0, 180], S et V dans [0, 255]
RED_RANGES = [((0, 120, 50), (10, 255, 255)), ((170, 120, 50), (180, 255, 255))]   # le rouge fait le tour
GREEN_RANGES = [((40, 100, 50), (85, 255, 255))]
MIN_AREA = 15          # pixels : en dessous, on considere que l'objet n'est pas vu
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def color_mask(rgb, ranges):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def largest_blob(mask):
    """Renvoie (contour, masque plein du contour, aire) du plus grand contour, ou None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, -1)
    return c, (filled > 0) & (mask > 0), area


def cube_center(pts, cam_pos, min_pts=6):
    """Centre du cube a partir des points 3D de sa surface visible.
    Hypothese : cube de taille connue, pose droit (pas tourne autour de z), comme dans l'env.
      - face du DESSUS = points a moins de TOP_BAND du plus haut : z du centre = dessus - demi-cote ;
      - faces de COTE = les autres points. La face la plus vue est celle tournee vers la camera :
        selon l'axe (x ou y) le plus aligne avec la direction camera -> cube, le centre est un
        demi-cote derriere cette face ; selon l'autre axe, il est au milieu des points.
    Si une seule des deux familles est visible (dessus cache par la main, ou cote cache), on fait
    avec ce qu'on a."""
    top = pts[pts[:, 2] > pts[:, 2].max() - TOP_BAND]
    side = pts[pts[:, 2] <= pts[:, 2].max() - TOP_BAND]
    if len(top) < min_pts and len(side) < min_pts:
        side = pts
    z = np.median(top[:, 2]) - CUBE_HALF if len(top) >= min_pts else np.median(side[:, 2])
    if len(side) >= min_pts:
        med = np.median(side, axis=0)
        d = med[:2] - cam_pos[:2]                    # direction horizontale camera -> cube
        k = int(np.argmax(np.abs(d)))                # axe de profondeur (face vue)
        xy = med[:2].copy()
        xy[k] += np.sign(d[k]) * CUBE_HALF           # centre = un demi-cote derriere la face vue
        if len(top) >= min_pts:
            xy[1 - k] = np.median(top[:, 1 - k])     # axe transverse : le dessus est plus complet
    else:
        xy = np.median(top[:, :2], axis=0)
    return np.array([xy[0], xy[1], z])


def locate(cam, rgb, depth, ranges, kind):
    """Detection + position 3D d'un objet. kind = 'cube' ou 'target'.
    Renvoie un dict (pos monde, centre pixel, contour, aire) ou None si l'objet n'est pas vu."""
    blob = largest_blob(color_mask(rgb, ranges))
    if blob is None:
        return None
    contour, inside, area = blob
    # on retire 1 pixel au bord : la profondeur y melange l'objet et le fond ("flying pixels")
    core = cv2.erode(inside.astype(np.uint8), KERNEL) > 0
    if core.sum() < 5:
        core = inside
    v, u = np.nonzero(core & (depth > 0))
    if len(u) < 5:
        return None
    pts = cam.pixels_to_world(u, v, depth[v, u])
    if kind == "cube":
        cam_pos, _ = cam.pose()
        pos = cube_center(pts, cam_pos)
    else:
        # marquage plat au sol, vu en biais : sa moitie proche occupe plus de pixels que sa moitie
        # lointaine (perspective), donc la mediane des points est tiree vers la camera. On prend le
        # milieu de l'etendue en x et en y (percentiles 2-98 pour ignorer les points aberrants).
        lo, hi = np.percentile(pts[:, :2], [2, 98], axis=0)
        xy = (lo + hi) / 2
        # la position utile est le centre du cube pose dessus : un demi-cote au-dessus de la table
        pos = np.array([xy[0], xy[1], np.median(pts[:, 2]) + CUBE_HALF - MARK_THICK])
    m = cv2.moments(contour)
    center_px = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else (u.mean(), v.mean())
    return {"pos": pos, "center_px": center_px, "contour": contour, "area": area}


class ObjectTracker:
    """Estime cube et cible a chaque image, avec deux regles simples quand la vue est mauvaise :
      - objet non detecte (cache par le bras) : on garde la derniere estimation ;
      - cible : elle ne bouge pas, mais le cube pose dessus la cache en partie et decale sa tache.
        On ne la met a jour que si sa tache fait au moins 90 % de la plus grande vue."""

    def __init__(self, cam):
        self.cam = cam
        self.reset()

    def reset(self):
        self.cube = self.target = None
        self.target_best_area = 0.0
        self.last = {}

    def update(self):
        rgb, depth = self.cam.capture()
        cube = locate(self.cam, rgb, depth, RED_RANGES, "cube")
        target = locate(self.cam, rgb, depth, GREEN_RANGES, "target")
        if cube is not None:
            self.cube = cube["pos"]
        if target is not None and target["area"] >= 0.9 * self.target_best_area:
            self.target = target["pos"]
            self.target_best_area = max(self.target_best_area, target["area"])
        self.last = {"rgb": rgb, "depth": depth, "cube": cube, "target": target}
        return self.cube, self.target


def estimated_obs(obs, cube, target):
    """Remplace dans l'observation 'etat' (19 valeurs) le cube et la cible par leurs estimations.
    Ce qui vient du robot lui-meme (position et vitesse de l'outil, ouverture de la pince) est garde :
    le vrai robot le connait par ses codeurs et son modele geometrique."""
    o = np.array(obs, dtype=np.float64, copy=True)
    tcp = o[0:3]
    o[7:10] = cube
    o[10:13] = cube - tcp
    o[13:16] = target
    o[16:19] = target - cube
    return o
