from ._array import array_api_compat, Array, cpu_only_compatibility

from typing import Tuple
from scipy.spatial.transform import Rotation


@cpu_only_compatibility
def euler_to_matrix(convention: str, euler_angles: Array, degrees=False) -> Array:
    return Rotation.from_euler(convention, euler_angles, degrees=degrees).as_matrix()


def get_transform_matrix(shape: Tuple[int, int, int], euler_angles: Array, translation: Array, convention: str="XZX", degrees: bool=False):
    """
    Returns the transformation matrix in pixel coordinates.
    The transformation is the composition of a rotation (defined by 3 euler angles), and a translation (defined by a translation vector).
    The rotation is made around the center of the volume.
    Params:
        shape: Tuple[int, int, int]
            shape of the image to be transformed. D, H, W
        euler_angles: np.ndarray of shape ((N), 3)
            𝛗, 𝛉, 𝛙. See convention to see how they are used.
        translation: np.ndarray of shape ((N), 3)
        convention: str
            Euler angles convention in scipy terms. See `scipy.spatial.transform.Rotation`.
            Default to 'XZX'

                   a-------------b       numpy coordinates of points:                    
                  /             /|        - a = (0, 0, 0)                                   
                 /             / |        - b = (0, 0, W-1)                                 
                c-------------+  |        - c = (0, H-1, 0)                                     
                |             |  |        - d = (D-1, H-1, 0)            If the convention 'XZX' is used:                               
                |             |  |        - e = (D-1, H-1, W-1)             - first, rotate by 𝛗 around the X-axis. The XYZ frame is also rotated!
                |             |  +                                          - then, rotate by 𝛉 around the Z-axis.
                |             X Y                                           - finally, rotate by 𝛙 around the X-axis.
                |             ↑↗                                                                                                                            
                d-----------Z←e   <-- reference frame used for rotations. The center of the rotation is at (D/2, H/2, W/2).
                                      

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
        H_rot = xp.zeros((4,4), **array_kwargs)
    elif len(euler_angles.shape) == 2:
        H_rot = xp.zeros((euler_angles.shape[0],4,4), **array_kwargs)
    H_rot[..., 3, 3] = 1.
    H_center = xp.asarray(H_rot, copy=True)
    H_center[..., :3, 3] = -center                        # 1. translation to (0,0,0)
    H_center[..., [0,1,2], [0,1,2]] = 1. # diag to 1
    H_rot[..., :3, :3] = rot                              # 2. rotation
    H_rot[..., :3, 3] = translation + center              # 3. translation to center of image. 4. translation
    
    #   4-3-2 <- 1
    H = H_rot @ H_center
    return H