import cv2
import numpy as np
from shapely.geometry import LinearRing
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _order_points(points, closed=True):
    points = points.copy()
    path = [points[0]]
    used = set()
    used.add(0)

    tree = cKDTree(points)

    for _ in range(len(points) - 1):
        dist, idxs = tree.query(path[-1], k=len(points))
        for idx in idxs:
            if idx not in used:
                path.append(points[idx])
                used.add(idx)
                break

    if closed:
        path.append(path[0])

    return np.array(path)


def parse_trackimg(img="racetrack.png", neg=False):
    a = cv2.threshold(cv2.imread(img, cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)[1]
    if neg:
        a = 255-a

    skel = skeletonize(a).astype(np.uint8)*255
    centerline = LinearRing(_order_points(np.argwhere(np.flipud(np.rot90(skel, 1)))))

    return a, centerline


def discretize_line(centerline: LinearRing, sectors: int = 40):
    return np.array([centerline.xy]).T[np.linspace(0, len(centerline.xy[0]) - 1, sectors).astype(int)].reshape(-1, 2)