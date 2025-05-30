from itertools import cycle

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    'morphological_chan_vese',
    'morphological_geodesic_active_contour',
    'inverse_gaussian_gradient',
    'circle_level_set',
    'checkerboard_level_set'
]

__version__ = (2, 1, 1)
__version_str__ = ".".join(map(str, __version__))


class _fcycle(object):

    def __init__(self, iterable):
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1


def sup_inf(u):
    """SI operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)


def inf_sup(u):
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)


_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)),  # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI


def _check_input(image, init_level_set):
    if image.ndim not in [2, 3]:
        raise ValueError("`image` must be a 2 or 3-dimensional array.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")


def _init_level_set(init_level_set, image_shape):
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'circle':
            res = circle_level_set(image_shape)
        elif init_level_set == 'ellipsoid':
            res = ellipsoid_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in "
                             "['checkerboard', 'circle', 'ellipsoid']")
    else:
        res = init_level_set
    return res


def circle_level_set(image_shape, center=None, radius=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid) ** 2, 0))
    res = np.int8(phi > 0)
    return res


def ellipsoid_level_set(image_shape, center=None, semi_axis=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if semi_axis is None:
        semi_axis = tuple(i / 2 for i in image_shape)

    if len(center) != len(image_shape):
        raise ValueError("`center` and `image_shape` must have the same length.")

    if len(semi_axis) != len(image_shape):
        raise ValueError("`semi_axis` and `image_shape` must have the same length.")

    if len(image_shape) == 2:
        xc, yc = center
        rx, ry = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y: ((x - xc) / rx) ** 2 +
                         ((y - yc) / ry) ** 2,
            image_shape, dtype=float)
    elif len(image_shape) == 3:
        xc, yc, zc = center
        rx, ry, rz = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y, z: ((x - xc) / rx) ** 2 +
                            ((y - yc) / ry) ** 2 +
                            ((z - zc) / rz) ** 2,
            image_shape, dtype=float)
    else:
        raise ValueError("`image_shape` must be a 2- or 3-tuple.")

    res = np.int8(phi > 0)
    return res


def checkerboard_level_set(image_shape, square_size=5):
    grid = np.ogrid[[slice(i) for i in image_shape]]
    grid = [(grid_i // square_size) & 1 for grid_i in grid]

    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def morphological_chan_vese(image, iterations, init_level_set='checkerboard',
                            smoothing=1, lambda1=1, lambda2=1,
                            iter_callback=lambda x: None):
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-8)
        c1 = (image * u).sum() / float(u.sum() + 1e-8)

        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c1) ** 2 - lambda2 * (image - c0) ** 2)

        u[aux < 0] = 1
        u[aux > 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u


def morphological_geodesic_active_contour(gimage, iterations,
                                          init_level_set='circle', smoothing=1,
                                          threshold='auto', balloon=0,
                                          iter_callback=lambda x: None):
    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 40)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u
