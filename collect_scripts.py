"""Script to collect and organize code files into a markdown document."""

from pathlib import Path
from typing import List, Tuple, Set


def gather_code_files(
    root_dir: Path, extensions: Set[str], exclude_files: Set[str], exclude_folders: Set[str]
) -> Tuple[List[Path], List[Path]]:
    """Gather code files while respecting exclusion rules."""
    try:
        code_files: List[Path] = []
        excluded_files_found: List[Path] = []

        for file_path in root_dir.rglob("*"):
            if any(excluded in file_path.parts for excluded in exclude_folders):
                if file_path.is_file():
                    excluded_files_found.append(file_path)
                continue

            if file_path.is_file():
                if file_path.name in exclude_files:
                    excluded_files_found.append(file_path)
                elif file_path.suffix in extensions:
                    code_files.append(file_path)

        return code_files, excluded_files_found
    except Exception as e:
        raise RuntimeError(f"Error gathering code files: {str(e)}")


def write_to_markdown(code_files: List[Path], excluded_files: List[Path], output_file: Path) -> None:
    """Write collected files to a markdown document."""
    try:
        with output_file.open("w", encoding="utf-8") as md_file:
            for file_path in code_files:
                relative_path = file_path.relative_to(file_path.cwd())
                md_file.write(f"## {relative_path}\n\n")
                md_file.write("```" + file_path.suffix.lstrip(".") + "\n")
                md_file.write(file_path.read_text(encoding="utf-8"))
                md_file.write("\n```\n\n")
    except Exception as e:
        raise RuntimeError(f"Error writing markdown file: {str(e)}")


def create_markdown(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str],
    output_file: Path = Path("code_base.md"),
) -> None:
    """Create a markdown file containing all code files."""
    try:
        code_files, excluded_files = gather_code_files(root_dir, extensions, exclude_files, exclude_folders)
        write_to_markdown(code_files, excluded_files, output_file)
        print(
            f"Markdown file '{output_file}' created with {len(code_files)} code files \
                and {len(excluded_files)} excluded files."
        )
    except Exception as e:
        raise RuntimeError(f"Error creating markdown: {str(e)}")


if __name__ == "__main__":
    root_directory = Path(__file__).parent
    extensions_to_look_for = {".py", ".ipynb", ".txt"}
    exclude_files_list = {".env", "__init__.py", "init.py"}
    exclude_folders_list = {"venv"}

    create_markdown(root_directory, extensions_to_look_for, exclude_files_list, exclude_folders_list)
