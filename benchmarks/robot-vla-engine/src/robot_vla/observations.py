"""LIBERO input conversion delegates camera and quaternion semantics to LeRobot."""

import numpy as np
import torch
from lerobot.processor.env_processor import LiberoProcessorStep


def _tensor_tree(value):
    if isinstance(value, dict):
        return {k: _tensor_tree(v) for k, v in value.items() if v is not None}
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32).unsqueeze(0)


def libero_batch(obs):
    batch = {
        f"observation.images.{key}": torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
        for key, image in obs["pixels"].items()
    }
    batch["observation.robot_state"] = _tensor_tree(obs["robot_state"])
    # Pinned upstream implementation performs both 180-degree image rotation and quaternion conversion.
    return LiberoProcessorStep()._process_observation(batch)
