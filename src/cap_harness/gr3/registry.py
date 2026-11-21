"""Typed discovery for the GR-3 cube-pick tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Gr3RegistryError(ValueError):
    """Raised when a GR-3 task reference is invalid."""


@dataclass(frozen=True, slots=True)
class Gr3TaskMetadata:
    task_name: str
    language: str
    target: str
    """Native name of the cube that must be lifted (never shown to programs)."""
    camera_names: tuple[str, ...] = ("rs_agentview", "rs_frontview")
    task_id: int = 0
    family: str = "gr3"
    horizon: int | None = 600
    lift_height_m: float = 0.10

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "camera_names": list(self.camera_names),
            "family": self.family,
            "language": self.language,
            "robots": ["gr3v2_1_1"],
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


GR3_TASKS: tuple[Gr3TaskMetadata, ...] = (
    Gr3TaskMetadata(
        "pick_red_cube",
        "Pick up the red cube and lift it clear of the table.",
        "cubeA",
    ),
    Gr3TaskMetadata(
        "pick_green_cube",
        "Pick up the green cube and lift it clear of the table.",
        "cubeB",
    ),
)


class Gr3TaskRegistry:
    """Resolve only the audited GR-3 task set."""

    def __init__(self, tasks: tuple[Gr3TaskMetadata, ...] = GR3_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise Gr3RegistryError("GR-3 task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[Gr3TaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self, task_ref: str | tuple[str, int] | Gr3TaskMetadata, task_id: int | None = None
    ) -> Gr3TaskMetadata:
        if isinstance(task_ref, Gr3TaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise Gr3RegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise Gr3RegistryError("conflicting GR-3 task ids")
            task_id = int(tuple_id)
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise Gr3RegistryError("conflicting GR-3 task ids")
                task_id = int(encoded_id)
        if task_id not in (None, 0):
            raise Gr3RegistryError("each GR-3 task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise Gr3RegistryError(
                f"unknown GR-3 task {task_name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc


__all__ = ["GR3_TASKS", "Gr3RegistryError", "Gr3TaskMetadata", "Gr3TaskRegistry"]
