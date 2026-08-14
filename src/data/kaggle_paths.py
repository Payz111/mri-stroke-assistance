"""Locate the training datasets inside a Kaggle input mount.

Kaggle mounts datasets at ``/kaggle/input/datasets/<owner>/<slug>/`` and the
archives nest unpredictably -- the ISLES export puts ``ISLES-2022`` inside
``ISLES-2022``, so its subject folders sit six levels below the input root.

This lives in the repository rather than inline in the notebook: notebook cells
are a copy held on Kaggle and do not update when the repo is cloned, so any fix
written inline has to be pasted in by hand and silently rots.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

DEFAULT_INPUT_ROOT = Path("/kaggle/input")
ISLES_PREFIX = "sub-strokecase"
SOOP_PREFIX = "sub-"


def has_subjects(path: Path, prefix: str) -> bool:
    """True if *path* directly contains directories starting with *prefix*."""
    try:
        return any(d.name.startswith(prefix) for d in path.iterdir() if d.is_dir())
    except (NotADirectoryError, PermissionError, FileNotFoundError, OSError):
        return False


def find_dataset_root(
    input_root: str | Path,
    prefix: str,
    reject: str | None = None,
    max_depth: int = 8,
) -> Path | None:
    """Breadth-first search for the directory that directly holds ``<prefix>*``.

    Parameters
    ----------
    input_root:
        Where to start searching, usually ``/kaggle/input``.
    prefix:
        Subject-directory prefix to look for.
    reject:
        If given, a candidate is skipped when it *also* holds directories with
        this prefix. Used to tell SOOP (``sub-1``) apart from ISLES
        (``sub-strokecase0001``) regardless of directory ordering.
    max_depth:
        Levels to descend. The real Kaggle layout needs six.

    Returns
    -------
    Path or None
    """
    frontier = [Path(input_root)]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for node in frontier:
            if has_subjects(node, prefix) and not (
                reject is not None and has_subjects(node, reject)
            ):
                return node
            try:
                for child in node.iterdir():
                    # Subject folders never contain a dataset root, and SOOP has
                    # 1323 of them -- descending into them is pure waste.
                    if child.is_dir() and not child.name.startswith("sub-"):
                        nxt.append(child)
            except (NotADirectoryError, PermissionError, FileNotFoundError, OSError):
                continue
        if not nxt:
            break
        frontier = nxt
    return None


def describe_tree(root: str | Path, max_depth: int = 7, _prefix: str = "") -> list[str]:
    """Render the input tree, marking directories that hold subject folders.

    The depth matches the search: ISLES subjects sit six levels below the input
    root, so a shallower listing would hide the very thing being looked for.
    """
    lines: list[str] = []
    if max_depth <= 0:
        return lines
    try:
        children = sorted(d for d in Path(root).iterdir() if d.is_dir())
    except (NotADirectoryError, PermissionError, FileNotFoundError, OSError):
        return lines
    for child in children[:8]:
        marker = ""
        if has_subjects(child, ISLES_PREFIX):
            marker = "   <- ISLES subjects here"
        elif has_subjects(child, SOOP_PREFIX):
            marker = "   <- SOOP subjects here"
        lines.append(f"{_prefix}  {child.name}/{marker}")
        if not child.name.startswith("sub-"):
            lines.extend(describe_tree(child, max_depth - 1, _prefix + "  "))
    return lines


def _fail(what: str, input_root: Path, hint: str) -> FileNotFoundError:
    tree = "\n".join(describe_tree(input_root)) or "  (nothing found)"
    return FileNotFoundError(f"{what} not found under {input_root}.\n\nInput tree:\n{tree}\n\n{hint}")


def locate_isles(input_root: str | Path = DEFAULT_INPUT_ROOT) -> tuple[Path, Path]:
    """Return ``(isles_root, derivatives_root)``.

    ``isles_root`` is the directory that directly contains ``sub-strokecase*``.
    Derivatives usually sit beside it, but some exports keep them a level up.
    """
    input_root = Path(input_root)
    root = find_dataset_root(input_root, ISLES_PREFIX)
    if root is None:
        raise _fail("ISLES-2022", input_root, "Attach the ISLES 2022 dataset in Kaggle settings.")

    derivatives = root / "derivatives"
    if not derivatives.is_dir():
        found = next((p for p in root.parents if (p / "derivatives").is_dir()), None)
        if found is not None:
            derivatives = found / "derivatives"
    return root, derivatives


def locate_soop(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    unpack_dir: str | Path = "/tmp/soop",
) -> Path:
    """Return the ds004889 root, unpacking the tar distribution if that is what is attached."""
    input_root = Path(input_root)
    unpack_dir = Path(unpack_dir)

    already = unpack_dir / "ds004889"
    if has_subjects(already, SOOP_PREFIX):
        return already

    root = find_dataset_root(input_root, SOOP_PREFIX, reject=ISLES_PREFIX)
    if root is not None:
        return root

    tar_path = next(Path(input_root).rglob("soop_ds004889.tar"), None)
    if tar_path is not None:
        os.makedirs(unpack_dir, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(unpack_dir)
        return unpack_dir / "ds004889"

    raise _fail(
        "SOOP (ds004889)",
        input_root,
        "Attach the ds004889 dataset, or the tar produced by notebooks/03a_download_soop.",
    )
