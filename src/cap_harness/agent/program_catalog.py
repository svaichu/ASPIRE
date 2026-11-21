"""Small, immutable catalog of reviewed programs shipped in the wheel."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from importlib.resources import files


@dataclass(frozen=True)
class CapProgramDefinition:
    name: str
    summary: str
    moves_robot: bool = True

    @property
    def tool_name(self) -> str:
        return f"cap_program_{self.name}"


@dataclass(frozen=True)
class LoadedCapProgram:
    definition: CapProgramDefinition
    source: str
    sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.definition.name,
            "summary": self.definition.summary,
            "moves_robot": self.definition.moves_robot,
            "sha256": self.sha256,
        }


YAM_PROGRAMS = (
    CapProgramDefinition("observe", "Read the current robot state.", False),
    CapProgramDefinition(
        "go_home",
        "Open both grippers, return both arms to their calibrated home pose, "
        "and report measured joint residuals.",
    ),
)


class CapProgramCatalog:
    """Expose fixed programs; remote callers cannot supply code or paths.

    Programs are ``<name>.py`` resources of ``package``; the default is the YAM
    System 2 catalog.
    """

    def __init__(
        self,
        definitions: Iterable[CapProgramDefinition] = YAM_PROGRAMS,
        *,
        package: str = "cap_harness.agent.programs",
    ) -> None:
        self._programs = {}
        for definition in definitions:
            source = files(package).joinpath(f"{definition.name}.py").read_text()
            self._programs[definition.name] = LoadedCapProgram(
                definition, source, hashlib.sha256(source.encode()).hexdigest()
            )

    def all(self) -> tuple[LoadedCapProgram, ...]:
        return tuple(self._programs.values())

    def get(self, name: str) -> LoadedCapProgram:
        return self._programs[name]

    def get_by_tool(self, name: str) -> LoadedCapProgram:
        return next(p for p in self.all() if p.definition.tool_name == name)
