"""Typed discovery for the ElRobot pizza tasks.

A sim task is a language goal plus an authored scene. ``scene_reset`` says how the
scene is restored between trials so a recorded run states it instead of implying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ElRobotRegistryError(ValueError):
    """Raised when an ElRobot task reference is invalid."""


@dataclass(frozen=True, slots=True)
class ElRobotTaskMetadata:
    task_name: str
    language: str
    task_id: int = 0
    family: str = "elrobot"
    scene_reset: str = "station_reset"

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "language": self.language,
            "scene_reset": self.scene_reset,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


ELROBOT_TASKS: tuple[ElRobotTaskMetadata, ...] = (
    ElRobotTaskMetadata("observe_station", "Observe the pizza station.", scene_reset="none"),
    ElRobotTaskMetadata("reach_home", "Return the arm to the home pose.", scene_reset="none"),
    ElRobotTaskMetadata(
        "pizza_assembly",
        "Place salami discs and cheese shreds from the bins onto the pizza plate.",
    ),
)


class ElRobotTaskRegistry:
    """Resolve only the explicitly supported ElRobot task set."""

    def __init__(self, tasks: tuple[ElRobotTaskMetadata, ...] = ELROBOT_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise ElRobotRegistryError("ElRobot task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[ElRobotTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self, task_ref: str | tuple[str, int] | ElRobotTaskMetadata, task_id: int | None = None
    ) -> ElRobotTaskMetadata:
        if isinstance(task_ref, ElRobotTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise ElRobotRegistryError("task tuple must contain (task_name, task_id)")
            task_name, task_id = task_ref[0], int(task_ref[1])
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded = task_name.rsplit(":", 1)
                task_id = int(encoded)
        if task_id not in (None, 0):
            raise ElRobotRegistryError("each ElRobot task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise ElRobotRegistryError(
                f"unknown ElRobot task {task_name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc


__all__ = [
    "ELROBOT_TASKS",
    "ElRobotRegistryError",
    "ElRobotTaskMetadata",
    "ElRobotTaskRegistry",
]
