
from typing import Optional
import torch

class LossBasedAdaptiveLRScheduler:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_early_max: float = 1e-3,
        lr_early_min: float = 2e-4,
        lr_mid_max: float = 2e-4,
        lr_mid_min: float = 1e-4,
        lr_late_fixed: float = 1e-4,
        patience: int = 500,
        improvement_threshold: float = 1e-6,
        lr_decay_factor: float = 0.5,
        verbose: bool = False,
        warmup_epochs: int = 100,
        early_ratio: float = 0.6,
        mid_ratio: float = 0.85,
        min_early_epochs: int = 1000,
        min_mid_epochs: int = 2000
    ):
        self.optimizer = optimizer
        self.lr_early_max = lr_early_max
        self.lr_early_min = lr_early_min
        self.lr_mid_max = lr_mid_max
        self.lr_mid_min = lr_mid_min
        self.lr_late_fixed = lr_late_fixed
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.lr_decay_factor = lr_decay_factor
        self.verbose = verbose

        self.warmup_epochs = warmup_epochs
        self.early_ratio = early_ratio
        self.mid_ratio = mid_ratio

        self.min_early_epochs = min_early_epochs
        self.min_mid_epochs = min_mid_epochs

        self.loss_history = []
        self.current_lr = lr_early_max
        self.current_phase = "early"
        self.step_count = 0

        self.initial_loss = None
        self.warmup_losses = []
        self.dynamic_threshold_early = None
        self.dynamic_threshold_mid = None
        self.warmup_complete = False

        self._set_lr(self.current_lr)

    def _calibrate_dynamic_thresholds(self) -> None:
        if len(self.warmup_losses) < self.warmup_epochs:
            return

        initial_loss = self.warmup_losses[0]
        min_warmup_loss = min(self.warmup_losses)
        loss_range = initial_loss - min_warmup_loss

        if abs(loss_range) > 1e-10:
            self.dynamic_threshold_early = initial_loss - loss_range * self.early_ratio
            self.dynamic_threshold_mid = initial_loss - loss_range * self.mid_ratio
        else:
            default_range = abs(initial_loss) if abs(initial_loss) > 1e-10 else 0.01
            self.dynamic_threshold_early = initial_loss - default_range * self.early_ratio
            self.dynamic_threshold_mid = initial_loss - default_range * self.mid_ratio
            if self.verbose:
                print(f"[LR Scheduler] Warning: loss_range ≈ 0, using default range: {default_range:.4e}")

        self.warmup_complete = True

        if self.verbose:
            print(f"[LR Scheduler] Dynamic thresholds calibrated after {self.warmup_epochs} epochs:")
            print(f"   Initial loss: {initial_loss:.4e}")
            print(f"   Min warmup loss: {min_warmup_loss:.4e}")
            print(f"   Loss range: {loss_range:.4e}")
            print(f"   Early threshold ({self.early_ratio*100:.0f}%): {self.dynamic_threshold_early:.4e}")
            print(f"   Mid threshold ({self.mid_ratio*100:.0f}%): {self.dynamic_threshold_mid:.4e}")
            print(f"   Min stage epochs: early≥{self.min_early_epochs}, mid≥{self.min_mid_epochs} (cumulative)")

    def _determine_phase_forward_only(self, current_loss: float) -> str:
        if not self.warmup_complete:
            return "early"

        threshold_early = self.dynamic_threshold_early
        threshold_mid = self.dynamic_threshold_mid

        if self.current_phase == "early":
            if self.step_count < self.min_early_epochs:
                return "early"
            if current_loss < threshold_mid and self.step_count >= self.min_mid_epochs:
                return "late"
            elif current_loss < threshold_early:
                return "mid"
            return "early"

        elif self.current_phase == "mid":
            if self.step_count < self.min_mid_epochs:
                return "mid"
            if current_loss < threshold_mid:
                return "late"
            return "mid"

        else:
            return "late"

    def step(self, current_loss: float) -> float:
        self.step_count += 1

        if self.initial_loss is None:
            self.initial_loss = current_loss

        if not self.warmup_complete:
            self.warmup_losses.append(current_loss)
            if len(self.warmup_losses) >= self.warmup_epochs:
                self._calibrate_dynamic_thresholds()

        phase = self._determine_phase_forward_only(current_loss)

        if phase != self.current_phase:
            if self.verbose:
                print(f"[LR Scheduler] Phase transition: {self.current_phase} → {phase} at step {self.step_count}")
            self.current_phase = phase
            self.loss_history = []

            if phase == "mid" and self.current_lr > self.lr_mid_max:
                if self.verbose:
                    print(f"[LR Scheduler] Clamping LR from {self.current_lr:.2e} to {self.lr_mid_max:.2e} (mid phase max)")
                self.current_lr = self.lr_mid_max
                self._set_lr(self.current_lr)
            elif phase == "late" and self.current_lr > self.lr_late_fixed:
                if self.verbose:
                    print(f"[LR Scheduler] Clamping LR from {self.current_lr:.2e} to {self.lr_late_fixed:.2e} (late phase fixed)")
                self.current_lr = self.lr_late_fixed
                self._set_lr(self.current_lr)

        if self.current_phase == "late":
            new_lr = self.lr_late_fixed
        else:
            new_lr = self._adaptive_adjustment(current_loss, self.current_phase)

        if new_lr != self.current_lr:
            self._set_lr(new_lr)
            if self.verbose:
                print(f"[LR Scheduler] Step {self.step_count}: lr changed {self.current_lr:.2e} → {new_lr:.2e}")
            self.current_lr = new_lr

        return new_lr

    def _adaptive_adjustment(self, current_loss: float, phase: str) -> float:
        self.loss_history.append(current_loss)

        if len(self.loss_history) > self.patience:
            self.loss_history.pop(0)

        if len(self.loss_history) < self.patience:
            return self.current_lr

        oldest_loss = self.loss_history[0]
        improvement_per_epoch = (oldest_loss - current_loss) / self.patience

        if phase == "early":
            lr_max = self.lr_early_max
            lr_min = self.lr_early_min
        else:
            lr_max = self.lr_mid_max
            lr_min = self.lr_mid_min

        if improvement_per_epoch < self.improvement_threshold:
            new_lr = self.current_lr * self.lr_decay_factor
            new_lr = max(new_lr, lr_min)
        else:
            new_lr = self.current_lr

        return new_lr

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.current_lr

    def get_phase(self) -> str:
        return self.current_phase

    def state_dict(self) -> dict:
        return {
            'loss_history': self.loss_history,
            'current_lr': self.current_lr,
            'current_phase': self.current_phase,
            'step_count': self.step_count,
            'initial_loss': self.initial_loss,
            'warmup_losses': self.warmup_losses,
            'dynamic_threshold_early': self.dynamic_threshold_early,
            'dynamic_threshold_mid': self.dynamic_threshold_mid,
            'warmup_complete': self.warmup_complete
        }

    def load_state_dict(self, state_dict: dict):
        self.loss_history = state_dict['loss_history']
        self.current_lr = state_dict['current_lr']
        self.current_phase = state_dict['current_phase']
        self.step_count = state_dict['step_count']
        self.initial_loss = state_dict.get('initial_loss', None)
        self.warmup_losses = state_dict.get('warmup_losses', [])
        self.dynamic_threshold_early = state_dict.get('dynamic_threshold_early', None)
        self.dynamic_threshold_mid = state_dict.get('dynamic_threshold_mid', None)
        self.warmup_complete = state_dict.get('warmup_complete', False)
        self._set_lr(self.current_lr)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    use_adaptive_lr: bool = True,
    **kwargs
) -> Optional[LossBasedAdaptiveLRScheduler]:
    if not use_adaptive_lr:
        return None

    return LossBasedAdaptiveLRScheduler(optimizer, **kwargs)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adaptive LR Scheduler v3.0 (Pure Dynamic Threshold)")
    print("=" * 60)

    import torch.nn as nn
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = LossBasedAdaptiveLRScheduler(
        optimizer,
        lr_early_max=1e-3,
        lr_early_min=2e-4,
        lr_mid_max=2e-4,
        lr_mid_min=1e-4,
        lr_late_fixed=1e-4,
        patience=10,
        improvement_threshold=1e-6,
        lr_decay_factor=0.5,
        verbose=True,
        warmup_epochs=5,
        early_ratio=0.2,
        mid_ratio=0.6
    )

    print("\n" + "=" * 60)
    print("Test 1: C-C boundary (loss starts high)")
    print("=" * 60)

    simulated_losses_cc = [
        0.1, 0.05, 0.02, 0.01, 0.005,
        0.0, -0.005, -0.008, -0.009, -0.0095,
        -0.010, -0.0105, -0.0108, -0.0110, -0.0112
    ]

    print("\nSimulated Training (C-C):")
    for epoch, loss in enumerate(simulated_losses_cc, 1):
        scheduler.step(loss)
        print(f"Epoch {epoch:2d}: loss={loss:+.5f}, lr={scheduler.get_lr():.2e}, phase={scheduler.get_phase()}")

    print("\n" + "=" * 60)
    print("Test 2: H-H boundary (loss starts low)")
    print("=" * 60)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler2 = LossBasedAdaptiveLRScheduler(
        optimizer2,
        lr_early_max=1e-3,
        lr_early_min=2e-4,
        lr_mid_max=2e-4,
        lr_mid_min=1e-4,
        lr_late_fixed=1e-4,
        patience=10,
        improvement_threshold=1e-6,
        lr_decay_factor=0.5,
        verbose=True,
        warmup_epochs=5,
        early_ratio=0.2,
        mid_ratio=0.6
    )

    simulated_losses_hh = [
        -0.00016, -0.00069, -0.00157, -0.00697, -0.00841,
        -0.00842, -0.00843, -0.00844, -0.00845, -0.00846,
        -0.00847, -0.00848, -0.00849, -0.00850, -0.00851
    ]

    print("\nSimulated Training (H-H):")
    for epoch, loss in enumerate(simulated_losses_hh, 1):
        scheduler2.step(loss)
        print(f"Epoch {epoch:2d}: loss={loss:+.5f}, lr={scheduler2.get_lr():.2e}, phase={scheduler2.get_phase()}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
