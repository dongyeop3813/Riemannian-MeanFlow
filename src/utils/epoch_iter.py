from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    ProgressColumn,
)
from rich.text import Text


class IterSpeedColumn(ProgressColumn):
    def render(self, task) -> Text:
        speed = task.speed
        if not speed:
            return Text("-- it/s")
        return Text(f"{speed:.2f} it/s")


class TrainingLoop:
    _instance = None

    # Singleton pattern
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create and remember the instance
            cls._instance = super().__new__(cls)
        return cls._instance

    def epoch_iter(self, start_epoch, end_epoch):
        columns = [
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=50),
            MofNCompleteColumn(),
            IterSpeedColumn(),
            TextColumn("Elapsed:", justify="right", style="bold"),
            TimeElapsedColumn(),
            TextColumn("Eta:", justify="right", style="bold"),
            TimeRemainingColumn(),
        ]

        with Progress(*columns) as self.progress:
            epoch_task = self.progress.add_task("Epoch", total=end_epoch - start_epoch)

            for epoch in range(start_epoch, end_epoch):
                yield epoch
                self.progress.update(epoch_task, advance=1)
            self.progress.remove_task(epoch_task)

    def train_iter(self, train_dataloader):
        return self.data_iter(train_dataloader, "Train")

    def val_iter(self, val_dataloader):
        return self.data_iter(val_dataloader, "Val")

    def data_iter(self, dataloader, description):
        if not hasattr(self, "progress"):
            raise ValueError("Epoch iter should be called before train or val iter")

        task = self.progress.add_task(description, total=len(dataloader))
        for batch in dataloader:
            yield batch
            self.progress.update(task, advance=1)
        self.progress.remove_task(task)


# Iterator for pretty printing of training loop
loop = TrainingLoop()
epoch_iter = loop.epoch_iter
train_iter = loop.train_iter
val_iter = loop.val_iter
