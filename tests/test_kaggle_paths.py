"""Tests for dataset discovery on a Kaggle input mount.

Reproduces the layout that broke two training attempts:
/kaggle/input/datasets/<owner>/<slug>/ISLES-2022/ISLES-2022/sub-strokecase0001
-- six levels down, one past the original search limit.
"""

from __future__ import annotations

import pytest

from src.data.kaggle_paths import (
    describe_tree,
    find_dataset_root,
    has_subjects,
    locate_isles,
    locate_soop,
)


def build_input(tmp_path, isles_owner="orvile", soop_owner="payzutdinmugutdinov"):
    """The real Kaggle mount: nested ISLES, SOOP as a plain dataset directory."""
    root = tmp_path / "input"

    isles = (
        root / "datasets" / isles_owner / "isles-2022-brain-stoke-dataset"
        / "ISLES-2022" / "ISLES-2022"
    )
    for i in (1, 2, 3):
        (isles / f"sub-strokecase{i:04d}" / "ses-0001" / "dwi").mkdir(parents=True)
    (isles / "derivatives").mkdir(parents=True)

    soop = root / "datasets" / soop_owner / "soop-ds" / "ds004889"
    for i in range(1, 30):
        (soop / f"sub-{i}" / "dwi").mkdir(parents=True)
    (soop / "derivatives" / "lesion_masks").mkdir(parents=True)

    return root, isles, soop


class TestFindDatasetRoot:
    def test_finds_deeply_nested_isles(self, tmp_path):
        root, isles, _ = build_input(tmp_path)
        assert find_dataset_root(root, "sub-strokecase") == isles

    def test_reject_separates_soop_from_isles(self, tmp_path):
        root, _, soop = build_input(tmp_path)
        assert find_dataset_root(root, "sub-", reject="sub-strokecase") == soop

    @pytest.mark.parametrize(
        ("isles_owner", "soop_owner"),
        [("aaa", "zzz"), ("zzz", "aaa")],
    )
    def test_result_does_not_depend_on_directory_order(self, tmp_path, isles_owner, soop_owner):
        """Neither dataset may be found merely because iterdir returned it first."""
        root, isles, soop = build_input(tmp_path, isles_owner, soop_owner)

        assert find_dataset_root(root, "sub-strokecase") == isles
        assert find_dataset_root(root, "sub-", reject="sub-strokecase") == soop

    def test_returns_none_when_absent(self, tmp_path):
        (tmp_path / "input" / "datasets" / "other").mkdir(parents=True)
        assert find_dataset_root(tmp_path / "input", "sub-strokecase") is None

    def test_respects_the_depth_limit(self, tmp_path):
        root, isles, _ = build_input(tmp_path)
        assert find_dataset_root(root, "sub-strokecase", max_depth=3) is None
        assert find_dataset_root(root, "sub-strokecase", max_depth=8) == isles

    def test_does_not_descend_into_subject_folders(self, tmp_path):
        """A decoy inside a subject folder must not be picked up."""
        root, isles, _ = build_input(tmp_path)
        decoy = isles / "sub-strokecase0001" / "nested" / "sub-strokecase9999"
        decoy.mkdir(parents=True)

        assert find_dataset_root(root, "sub-strokecase") == isles


class TestLocate:
    def test_locate_isles_returns_root_and_derivatives(self, tmp_path):
        root, isles, _ = build_input(tmp_path)

        found_root, derivatives = locate_isles(root)

        assert found_root == isles
        assert derivatives == isles / "derivatives"

    def test_derivatives_found_one_level_up(self, tmp_path):
        root, isles, _ = build_input(tmp_path)
        (isles / "derivatives").rmdir()
        (isles.parent / "derivatives").mkdir()

        _, derivatives = locate_isles(root)

        assert derivatives == isles.parent / "derivatives"

    def test_locate_soop(self, tmp_path):
        root, _, soop = build_input(tmp_path)
        assert locate_soop(root, unpack_dir=tmp_path / "unpack") == soop

    def test_locate_soop_prefers_already_unpacked(self, tmp_path):
        root, _, _ = build_input(tmp_path)
        unpack = tmp_path / "unpack"
        (unpack / "ds004889" / "sub-1").mkdir(parents=True)

        assert locate_soop(root, unpack_dir=unpack) == unpack / "ds004889"

    def test_missing_isles_raises_with_a_readable_tree(self, tmp_path):
        empty = tmp_path / "input"
        (empty / "datasets" / "someone" / "unrelated").mkdir(parents=True)

        with pytest.raises(FileNotFoundError) as excinfo:
            locate_isles(empty)

        message = str(excinfo.value)
        assert "Input tree:" in message
        assert "unrelated" in message

    def test_missing_soop_raises(self, tmp_path):
        empty = tmp_path / "input"
        empty.mkdir()

        with pytest.raises(FileNotFoundError) as excinfo:
            locate_soop(empty, unpack_dir=tmp_path / "unpack")

        assert "ds004889" in str(excinfo.value)


class TestHelpers:
    def test_has_subjects(self, tmp_path):
        root, isles, _ = build_input(tmp_path)
        assert has_subjects(isles, "sub-strokecase")
        assert not has_subjects(isles, "sub-999")

    def test_has_subjects_tolerates_a_file(self, tmp_path):
        target = tmp_path / "a_file.txt"
        target.write_text("x", encoding="utf-8")
        assert not has_subjects(target, "sub-")

    def test_describe_tree_marks_where_subjects_live(self, tmp_path):
        root, _, _ = build_input(tmp_path)

        rendered = "\n".join(describe_tree(root))

        assert "ISLES subjects here" in rendered
        assert "SOOP subjects here" in rendered
