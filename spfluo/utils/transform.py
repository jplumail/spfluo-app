from ._array import array_api_compat, Array, cpu_only_compatibility

from typing import Tuple
from scipy.spatial.transform import Rotation


@cpu_only_compatibility
def euler_to_matrix(convention: str, euler_angles: Array, degrees=False) -> Array:
    return Rotation.from_euler(convention, euler_angles, degrees=degrees).as_matrix()


def get_transform_matrix(shape: Tuple[int, int, int], euler_angles: Array, translation: Array, convention: str="ZXZ", degrees: bool=False):
    """
    Params:
        shape: Tuple[int, int, int]
            shape of the image to be transformed
        euler_angles: np.ndarray of shape ((N), 3)
        translation: np.ndarray of shape ((N), 3)
        convention: str
            Euler convention in scipy terms. See `scipy.spatial.transform.Rotation`.
        degrees: bool
            Are the euler angles in degrees?
    Returns:
        np.ndarray of shape ((N), 4, 4)
        An (or N) affine tranformation(s) in homogeneous coordinates.
    """
    xp = array_api_compat.array_namespace(euler_angles, translation)
    array_kwargs = {'dtype': euler_angles.dtype, 'device': xp.device(euler_angles)}
    rot = euler_to_matrix(convention, euler_angles, degrees=degrees)
    center = (xp.asarray(shape, **array_kwargs)-1) / 2
    if len(euler_angles.shape) == 1:
        H_rot = xp.zeros((4,4))
    elif len(euler_angles.shape) == 2:
        H_rot = xp.zeros((euler_angles.shape[0],4,4))
    H_rot[..., 3, 3] = 1.
    H_center = xp.asarray(H_rot, copy=True)
    H_center[..., :3, 3] = -center                        # 1. translation to (0,0,0)
    H_center[..., [0,1,2], [0,1,2]] = 1.
    H_rot[..., :3, :3] = rot                              # 2. rotation
    H_rot[..., :3, 3] = translation + center              # 3. translation to center of image. 4. translation
    
    #   2-3-4 <- 1
    H = H_rot @ H_center
    return H