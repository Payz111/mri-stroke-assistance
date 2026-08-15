"""Tests for the wall-clock budget in Trainer.fit.

A 12-hour Kaggle run was SIGKILLed at epoch 42 of 100. The best checkpoint
survived only because CheckpointCallback writes every improvement; the curves,
the metadata and the flat copies were all lost because the cells after fit()
never ran. The budget exists so training ends by itself, in time to save.
"""

from __future__ import annotations

import itertools

import pytest

from src.train.trainer import Trainer


class FakeTrainer(Trainer):
    """Trainer with the compute replaced by a clock we control."""

    def __init__(self, epoch_seconds, dice_sequence=None, **kwargs):
        # Bypass Trainer.__init__: none of the torch machinery is exercised here.
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = []
        self.val_loader = []
        self.device = "cpu"
        self.scheduler = None
        self.callbacks = []
        self.config = {}
        self.history = []
        self.use_amp = False
        self.max_hours = kwargs.get("max_hours")

        self._epoch_seconds = epoch_seconds
        self._dice = itertools.cycle(dice_sequence or [0.5])
        self._now = 0.0

    # Simulated clock, advanced only by train_epoch
    def _time(self):
        return self._now

    def train_epoch(self, epoch):
        self._now += self._epoch_seconds
        return {"train_loss": 0.1, "train_dice": 0.8}

    def validate_epoch(self, epoch):
        return {"val_loss": 0.1, "val_dice": next(self._dice)}


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """Make time.time() inside trainer read the simulated clock."""
    import src.train.trainer as trainer_module

    holder = {"trainer": None}

    def fake_time():
        instance = holder["trainer"]
        return instance._now if instance is not None else 0.0

    monkeypatch.setattr(trainer_module.time, "time", fake_time)
    return holder


def run(fake_clock, epoch_seconds, num_epochs, max_hours=None, dice=None):
    trainer = FakeTrainer(epoch_seconds, dice_sequence=dice, max_hours=max_hours)
    fake_clock["trainer"] = trainer
    return trainer.fit(num_epochs=num_epochs)


class TestWallClockBudget:
    def test_runs_all_epochs_when_they_fit(self, fake_clock):
        result = run(fake_clock, epoch_seconds=600, num_epochs=10, max_hours=11.0)

        assert result["total_epochs"] == 10
        assert result["stopped_on_time_budget"] is False

    def test_stops_before_the_budget_is_exhausted(self, fake_clock):
        """1000s epochs against an 11h budget: stop around 39, never at 100."""
        result = run(fake_clock, epoch_seconds=1000, num_epochs=100, max_hours=11.0)

        assert result["stopped_on_time_budget"] is True
        assert 35 <= result["total_epochs"] <= 40

    @pytest.mark.parametrize("epoch_seconds", [300, 1000, 2400])
    def test_leaves_the_save_margin_free(self, fake_clock, epoch_seconds):
        """Whatever the epoch length, SAVE_MARGIN_SECONDS must remain on exit."""
        budget_hours = 11.0
        result = run(
            fake_clock, epoch_seconds=epoch_seconds, num_epochs=1000, max_hours=budget_hours
        )

        spent = result["total_epochs"] * epoch_seconds
        remaining = budget_hours * 3600 - spent

        assert result["stopped_on_time_budget"] is True
        assert remaining >= Trainer.SAVE_MARGIN_SECONDS

    def test_no_budget_means_no_time_stop(self, fake_clock):
        result = run(fake_clock, epoch_seconds=100_000, num_epochs=3, max_hours=None)

        assert result["total_epochs"] == 3
        assert result["stopped_on_time_budget"] is False

    def test_slow_epochs_still_produce_a_result(self, fake_clock):
        """An epoch longer than the whole budget must not loop forever."""
        result = run(fake_clock, epoch_seconds=50_000, num_epochs=100, max_hours=1.0)

        assert result["total_epochs"] == 1
        assert result["stopped_on_time_budget"] is True

    def test_history_is_complete_for_the_epochs_that_ran(self, fake_clock):
        result = run(fake_clock, epoch_seconds=1000, num_epochs=100, max_hours=3.0)

        assert len(result["history"]) == result["total_epochs"]
        assert all("val_dice" in entry for entry in result["history"])

    def test_best_dice_is_tracked_across_the_truncated_run(self, fake_clock):
        result = run(
            fake_clock,
            epoch_seconds=1000,
            num_epochs=100,
            max_hours=2.0,
            dice=[0.40, 0.80, 0.55],
        )

        assert result["best_val_dice"] == pytest.approx(0.80)
        assert result["best_epoch"] == 1
