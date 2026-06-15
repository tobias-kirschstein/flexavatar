from typing import List, Callable

import numpy as np
import torch


def interpolate_codes(latent_codes: List[torch.Tensor], n_frames: int, loop: bool = False, ease_fn: Callable[[float], float] = lambda alpha: alpha) -> List[torch.Tensor]:
    n_codes = len(latent_codes)
    interpolation_i_codes = list(range(n_codes))  # Just interpolate between first n codes
    if loop:
        interpolation_i_codes.append(0)
        n_codes += 1
    interpolation_target_frames = np.linspace(0, n_frames - 1, n_codes, dtype=int)
    # interpolation_target_frames:  0| 11| 22| 33|
    # interpolation_phase:           | 0 | 1 | 2 |
    #
    interpolated_codes = []
    for frame_id in range(n_frames):
        interpolation_phase = (frame_id > interpolation_target_frames).sum() - 1

        if frame_id == 0:
            i_code_1 = interpolation_i_codes[0]
            i_code_2 = interpolation_i_codes[0]
            alpha = 1
        else:
            frame_id_1 = interpolation_target_frames[interpolation_phase]
            frame_id_2 = interpolation_target_frames[interpolation_phase + 1]
            i_code_1 = interpolation_i_codes[interpolation_phase]
            i_code_2 = interpolation_i_codes[interpolation_phase + 1]

            alpha = (frame_id - frame_id_1) / (frame_id_2 - frame_id_1)
            alpha = alpha.astype(latent_codes[0].dtype)

        alpha = ease_fn(alpha)

        code_1 = latent_codes[i_code_1]
        code_2 = latent_codes[i_code_2]

        interpolated_code = (1 - alpha) * code_1 + alpha * code_2
        interpolated_codes.append(interpolated_code)

    return interpolated_codes
