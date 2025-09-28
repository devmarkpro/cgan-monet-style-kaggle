import wandb
from dataclasses import asdict

from configs import AppParams


class MonetWandb:
    def __init__(self, project_name: str, params: AppParams):
        self.project_name = project_name
        self.params = params
        self._setup()

    def _setup(self):
        self.run = wandb.init(
            # Set the wandb project where this run will be logged.
            project=self.project_name,
            # Track hyperparameters and run metadata.
            config=asdict(self.params),
        )
