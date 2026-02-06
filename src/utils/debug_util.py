import numpy as np
from pathlib import Path
from .checkpoint import save_ckpt


class DebugLogger:
    def __init__(self, run_dir):
        self.debug_dir = Path(run_dir) / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Open the log file once in append mode and keep it open
        self.log_file = open(
            self.debug_dir / "log.txt", "a", buffering=1
        )  # line-buffered

    def loss_grad_and_loss(self, epoch, loss, grad_norm):
        """
        Write loss and grad_norm information for the given epoch to the log file.
        Keeps the file open for efficiency during frequent writes.
        """
        self.log_file.write(f"Epoch {epoch}: loss = {loss}, grad_norm = {grad_norm}\n")

    def __del__(self):
        # Ensure the log file is closed when the object is deleted
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()

    def checkpoint_and_exit(self, epoch, model, optimizer, **kwargs):
        np.save(
            self.debug_dir / f"loss_explode_{epoch}.npy",
            kwargs,
        )
        save_ckpt(self.debug_dir, model, optimizer, epoch)
        raise ValueError("Loss is too high")
