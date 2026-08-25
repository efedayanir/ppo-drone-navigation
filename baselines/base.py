from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from env.drone_env import DroneNavEnv


class BaselinePolicy(ABC):
    """Common interface for non-learning baseline controllers."""

    name = "base"

    @abstractmethod
    def act(
        self,
        env: DroneNavEnv,
        obs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return an action in the environment action-space format."""
        raise NotImplementedError
