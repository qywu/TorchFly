from genericpath import exists
from typing import Any, Dict
import os
import sys
import math
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from omegaconf import DictConfig

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)

Trainer = Any


@Callback.register("evaluation")
class Evaluation(Callback):
    """
    Callback that handles Evaluation and saves best model weights
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.config = config.evaluation
        self.enabled = config.evaluation.enabled
        self.last_save_time = time.time()
        self.started = False

    @handle_event(Events.TRAIN_BEGIN, priority=180)
    def setup_evaluation(self, trainer: Trainer):
        self.saved_top_k_models = []
        self.best_model_name = None

        # Checkpoint in seconds or steps
        if self.config.steps_interval > 0:
            self.evaluation_in_seconds = False
        else:
            if (not hasattr(self.config, "seconds_interval")) or self.config.seconds_interval < 0:
                self.evaluation_in_seconds = False
            else:
                self.evaluation_in_seconds = True

        self.tensorboard = trainer.tensorboard

        # evaluation steps interval
        if self.config.after_num_steps is None:
            self.evaluation_after_num_steps = 0
        else:
            self.evaluation_after_num_steps = self.config.after_num_steps

        if self.config.steps_interval < 0 and self.config.seconds_interval < 0:
            self.evaluate_in_epoch = trainer.training_in_epoch
            if self.evaluate_in_epoch == False:
                raise ValueError("Please set either `self.config.steps_interval` or `self.config.seconds_interval`")
        else:
            self.evaluate_in_epoch = False

    @handle_event(Events.TRAIN_BEGIN)
    def on_train_begin(self, trainer):
        # Start validation at the begining
        if not self.started:
            self._evaluation(trainer)
            self.started = True

    @handle_event(Events.TRAIN_END)
    def test_on_train_end(self, trainer):
        # Start validation at the begining
        if trainer.test_dataloader is not None and self.enabled:
            try:
                state_dict = torch.load(f"evaluation/{trainer.name}_model_weights/best.pth")
                trainer.model.load_state_dict(state_dict["model_weights"])
                logger.info(
                    f"Loading the `best.pth` epoch {state_dict['epochs_trained']} step {state_dict['global_step_count']} with score {state_dict['score']}!"
                )
            except:
                logger.warn("Cannot load `best.pth`!")

            trainer.test(trainer.test_dataloader)

    @handle_event(Events.STEP_END)
    def on_batch_end(self, trainer: Trainer):
        if not self.evaluate_in_epoch:
            # Check evaluation
            if trainer.global_step_count > self.evaluation_after_num_steps:
                if self.evaluation_in_seconds:
                    current_time = time.time()
                    # the elapsed time is longer than the seconds
                    if (current_time - self.last_save_time) > self.config.seconds_interval:
                        self._evaluation(trainer)
                        self.last_save_time = current_time
                else:
                    if (trainer.global_step_count + 1) % self.config.steps_interval == 0:
                        self._evaluation(trainer)

    @handle_event(Events.EPOCH_END)
    def on_epoch_end(self, trainer: Trainer):
        if self.evaluate_in_epoch:
            self._evaluation(trainer)

    @handle_event(Events.VALIDATE_BEGIN)
    def info_valid_begin(self, trainer: Trainer):
        logger.info(f"Validation starts at epoch {trainer.epochs_trained + 1} steps {trainer.global_step_count}")
        self.eval_start_time = time.time()
        # save model here
        os.makedirs("evaluation", exist_ok=True)
        os.makedirs(f"evaluation/{trainer.name}_model_weights", exist_ok=True)

    @handle_event(Events.VALIDATE_END)
    def record_validation_metrics(self, trainer: Trainer):
        log_string = f"Validation at epoch {trainer.epochs_trained + 1} steps {trainer.global_step_count} | duration {time.time() - self.eval_start_time:3.2f}s"
        metrics = trainer.model.get_evaluation_metrics()
        model_score = metrics["score"][1] if isinstance(metrics["score"], tuple) else metrics["score"]
        metrics_dict = {}
        # loop over all the metrics
        for metric_name, value in metrics.items():
            # if value is tuple, parse it
            if isinstance(value, tuple):
                display_value, value = value
            else:
                display_value = value
            log_string += f" | {metric_name} {display_value}"
            # tensorboard
            try:
                value = float(value)
                self.tensorboard.add_scalar("validate/" + metric_name, value, global_step=trainer.global_step_count)
            except:
                logger.warn("tensorboard is not functioning!")
            metrics_dict[metric_name] = value
        logger.info(log_string)

        # save the best model
        is_best = False
        if trainer.global_step_count > 0:
            if len(self.saved_top_k_models) > 0:
                self.saved_top_k_models = sorted(self.saved_top_k_models, key=lambda item: item["score"])

            if len(self.saved_top_k_models) > 0 and model_score > self.saved_top_k_models[-1]["score"]:
                model_path = os.path.join(f"evaluation/{trainer.name}_model_weights", f"best.pth")
                torch.save(self.get_model_weights_stamp(trainer, model_score), model_path)
                is_best = True

            if len(self.saved_top_k_models) < self.config.save_top_k_models:
                # save the model
                model_path = "epoch_" + str(trainer.epochs_trained) + "_step_" + str(trainer.global_step_count) + ".pth"
                model_path = os.path.join(f"evaluation/{trainer.name}_model_weights", model_path)
                torch.save(self.get_model_weights_stamp(trainer, model_score), model_path)
                self.saved_top_k_models.append({"path": model_path, "score": model_score})
            else:
                model_path = self.saved_top_k_models.pop(0)["path"]
                os.remove(model_path)

        with open("evaluation/results.txt", "a") as f:
            f.write(f"{trainer.name} validation @epoch {trainer.epochs_trained} @step {trainer.global_step_count} | ")
            f.write(json.dumps(metrics_dict))
            f.write(" | ")
            if is_best:
                f.write(" best so far")
            f.write("\n")

    @handle_event(Events.TEST_BEGIN)
    def info_test_begin(self, trainer: Trainer):
        logger.info(f"Test starts! ")
        os.makedirs("evaluation", exist_ok=True)
        self.eval_start_time = time.time()

    @handle_event(Events.TEST_END)
    def record_test_metrics(self, trainer: Trainer):
        log_string = f"Test"
        metrics = trainer.model.get_evaluation_metrics()
        metrics_dict = {}
        # loop over all the metrics
        for metric_name, value in metrics.items():
            # if value is tuple, parse it
            if isinstance(value, tuple):
                display_value, value = value
            else:
                display_value = value
            log_string += f" | {metric_name} {display_value}"
            # tensorboard
            try:
                value = float(value)
                self.tensorboard.add_scalar("test/" + metric_name, value, global_step=trainer.global_step_count)
            except:
                logger.warn("tensorboard is not functioning!")
            metrics_dict[metric_name] = value
        logger.info(log_string)

        with open("evaluation/results.txt", "a") as f:
            f.write(f"{trainer.name} test: ")
            f.write(json.dumps(metrics_dict))
            f.write("\n")

    def _evaluation(self, trainer):
        if trainer.validation_dataloader is not None and self.enabled:
            trainer.validate(trainer.validation_dataloader)

    def get_model_weights_stamp(self, trainer: Trainer, score: float):
        item = {
            "model_weights": trainer.model.state_dict(),
            "global_step_count": trainer.global_step_count,
            "epochs_trained": trainer.epochs_trained,
            "score": score
        }
        return item

    def state_dict(self):
        state_dict = {"started": self.started}
        return state_dict

    def load_state_dict(self, state_dict):
        self.started = state_dict["started"]