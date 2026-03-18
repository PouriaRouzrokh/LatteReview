"""Skill discovery and lazy loading from filesystem."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from lattereview.agentic.skills.base import SkillManifest

logger = logging.getLogger(__name__)

# Paths
BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin"


def parse_skill_md(skill_dir: Path) -> Optional[SkillManifest]:
    """Parse a SKILL.md file and return a SkillManifest.

    Extracts YAML frontmatter (name, description) and markdown body.
    Returns None if the directory doesn't contain a valid SKILL.md.

    Args:
        skill_dir: Path to the skill directory containing SKILL.md.

    Returns:
        SkillManifest with metadata and body, or None if invalid.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        logger.debug(f"No SKILL.md found in {skill_dir}")
        return None

    text = skill_md.read_text(encoding="utf-8")

    # Parse YAML frontmatter (between --- delimiters)
    if not text.startswith("---"):
        logger.warning(f"SKILL.md in {skill_dir} missing YAML frontmatter")
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning(f"SKILL.md in {skill_dir} has malformed frontmatter")
        return None

    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        logger.warning(f"SKILL.md in {skill_dir} has invalid YAML: {e}")
        return None

    if not isinstance(frontmatter, dict):
        logger.warning(f"SKILL.md in {skill_dir} frontmatter is not a dict")
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not name or not description:
        logger.warning(f"SKILL.md in {skill_dir} missing 'name' or 'description' in frontmatter")
        return None

    body = parts[2].strip()

    return SkillManifest(name=name, description=description, path=skill_dir, body=body)


def discover_skills(*search_paths: Path) -> list[SkillManifest]:
    """Discover skills from builtin and custom directories.

    Scans each directory for subdirectories containing a SKILL.md file.
    Builtin skills are always included. Custom paths are appended.

    Args:
        *search_paths: Additional directories to scan for skill folders.

    Returns:
        List of SkillManifest objects (L1 metadata only, toolsets not loaded).
    """
    manifests: list[SkillManifest] = []
    seen_names: set[str] = set()

    all_paths = [BUILTIN_SKILLS_DIR] + list(search_paths)

    for base_dir in all_paths:
        if not base_dir.exists() or not base_dir.is_dir():
            logger.debug(f"Skill search path does not exist: {base_dir}")
            continue

        for entry in sorted(base_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue

            manifest = parse_skill_md(entry)
            if manifest is None:
                continue

            if manifest.name in seen_names:
                logger.warning(f"Duplicate skill name '{manifest.name}' in {entry}, skipping")
                continue

            seen_names.add(manifest.name)
            manifests.append(manifest)

    return manifests


def load_toolset(manifest: SkillManifest) -> object:
    """Load a skill's FunctionToolset from its tools.py module.

    Uses importlib to dynamically import the tools.py file and extract
    the module-level `toolset` variable (a FunctionToolset instance).

    Args:
        manifest: SkillManifest with path to the skill directory.

    Returns:
        The FunctionToolset object from the skill's tools.py.

    Raises:
        FileNotFoundError: If tools.py doesn't exist.
        AttributeError: If tools.py doesn't export a `toolset` variable.
    """
    tools_py = manifest.path / "tools.py"
    if not tools_py.exists():
        raise FileNotFoundError(f"Skill '{manifest.name}' has no tools.py at {tools_py}")

    # Dynamic import using importlib
    module_name = f"lattereview.agentic.skills._dynamic_.{manifest.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, tools_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {tools_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    toolset = getattr(module, "toolset", None)
    if toolset is None:
        raise AttributeError(
            f"Skill '{manifest.name}' tools.py must export a module-level `toolset` variable "
            f"(a pydantic_ai.toolsets.FunctionToolset instance)"
        )

    return toolset
