from dataclasses import dataclass

import numpy as np
from dreifus.vector import Vec2


@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int

    def get_position(self) -> Vec2:
        return Vec2(self.x, self.y, dtype=int)

    def get_pos1(self) -> Vec2:
        return self.get_position()

    def get_pos2(self) -> Vec2:
        return Vec2(self.x + self.w - 1, self.y + self.h - 1, dtype=int)

    def get_dimensions(self) -> Vec2:
        return Vec2(self.w, self.h, dtype=int)

    def scale(self, scale: Vec2):
        # NB: Python's round() does round-to-even! Regular round can be implemented by int(x + 0.5)
        self.x = int(scale.x * self.x + 0.5)
        self.y = int(scale.y * self.y + 0.5)
        self.w = int(scale.x * self.w + 0.5)
        self.h = int(scale.y * self.h + 0.5)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y + self.h, self.x:self.x + self.w]
