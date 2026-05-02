from dataclasses import dataclass


@dataclass
class Pose:
    y: int
    x: int

    def __add__(self, o: "Pose") -> "Pose":
        return Pose(self.y + o.y, self.x + o.x)

    def __sub__(self, o: "Pose") -> "Pose":
        return Pose(self.y - o.y, self.x - o.x)

    def __floordiv__(self, k: int) -> "Pose":
        return Pose(self.y // k, self.x // k)

    def __mul__(self, k: int) -> "Pose":
        return Pose(self.y * k, self.x * k)
