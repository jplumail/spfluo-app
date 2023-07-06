from typing import Callable, Tuple

import albumentations as A

from . import volumentations as V

# +------------------------------------------------------------------------------------------+ #
# |                                      AUGMENTATION API                                    | #
# +------------------------------------------------------------------------------------------+ #


def augment_3d(patch_size: Tuple[int], p: float) -> V.Compose:
    return V.Compose(
        [
            V.RandomScale((0.8, 1.2), always_apply=True),
            V.PadIfNeeded(patch_size, always_apply=True),
            V.Resize(patch_size, always_apply=True),
            V.RandomRotate90((1, 2)),
            V.Flip(0),
        ],
        p=p,
    )


def augment_2d(patch_size: Tuple[int], p: float) -> Callable:
    size = max(patch_size)  # albumentations doesn't handle non square image.
    return A.Compose(
        [
            A.RandomRotate90(),
            A.Flip(),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5
            ),
            A.RandomSizedCrop((size - size // 5, size), size, size, p=0.8),
            # A.CoarseDropout(max_holes=5, max_height=8, max_width=8, p=0.5)
        ],
        p=p,
    )


def get_augment_policy(
    patch_size: Tuple[int], p: float = 0.8, dim: int = 2
) -> Callable:
    return augment_2d(patch_size, p) if dim == 2 else augment_3d(patch_size, p)
