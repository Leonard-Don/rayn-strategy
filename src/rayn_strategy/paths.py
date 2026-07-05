"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIRS = (
    Path("configs/active"),
    Path("configs/research"),
    Path("configs/baselines"),
    Path("configs/archive"),
)

REPORT_INPUT_DIRS = (
    Path("reports/decision-records"),
    Path("reports/evidence"),
    Path("reports/runtime"),
)


def resolve_config_path(path: str | Path) -> Path:
    """Resolve current and legacy config paths.

    Older commands used root-level names such as ``config.optimized.toml``.
    The repository now stores configs under ``configs/`` while keeping those
    legacy names resolvable for CLI users and research scripts.
    """

    candidate = Path(path)
    for direct in _direct_candidates(candidate):
        if direct.exists():
            return direct

    name = candidate.name
    if name.startswith("config.") and name.endswith(".toml"):
        for base in _search_bases(candidate):
            for config_dir in CONFIG_DIRS:
                resolved = base / config_dir / name
                if resolved.exists():
                    return resolved

    return candidate


def resolve_report_input_path(path: str | Path) -> Path:
    """Resolve existing report inputs from the new report subdirectories."""

    candidate = Path(path)
    for direct in _direct_candidates(candidate):
        if direct.exists():
            return direct

    report_parts = _repo_relative_parts(candidate)
    if len(report_parts) >= 2 and report_parts[0] == "reports":
        name = candidate.name
        for base in _search_bases(candidate):
            for report_dir in REPORT_INPUT_DIRS:
                resolved = base / report_dir / name
                if resolved.exists():
                    return resolved

    return candidate


def repo_output_path(path: str | Path) -> Path:
    """Return an absolute path inside the repo unless the input is absolute."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _direct_candidates(candidate: Path) -> list[Path]:
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(REPO_ROOT / candidate)
    return candidates


def _search_bases(candidate: Path) -> list[Path]:
    if candidate.is_absolute():
        return [REPO_ROOT]
    return [Path.cwd(), REPO_ROOT]


def _repo_relative_parts(candidate: Path) -> tuple[str, ...]:
    try:
        return candidate.relative_to(REPO_ROOT).parts
    except ValueError:
        return candidate.parts
