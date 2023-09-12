from typing import Tuple

import wandb
from tango import Step


@Step.register("log-metrics")
class LogMetrics(Step):
    VERSION = "003"

    def run(self, model_name, task_set, task, metrics: Tuple):  # , project: str, entity: str):
        # run_id: str = (
        #     wandb.run.id  # type: ignore[attr-defined]
        #     if wandb.run is not None
        #     else model_task_name
        # )

        # if wandb.run is None:
        #     wandb.init(
        #         id=run_id,
        #         dir=str(self.work_dir),
        #         project=project,
        #         entity=entity,
        #         job_type="visualize_metrics",
        #     )
        # else:
        #     pass

        full_metrics = {}
        metric_dict = metrics[0]
        for key in metric_dict:
            value = metric_dict[key][metric_dict[key]["primary_metric"]]
            full_metrics[model_name + "_" + task_set + "_" + task + "_" + key] = value
            wandb.log(full_metrics)
