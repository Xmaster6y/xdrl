from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs import ExplorationType
from torchrl.envs.utils import set_exploration_type
from torchrl.trainers.trainers import Trainer, TrainerHookBase


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _min_mean_max(prefix: str, value: torch.Tensor) -> dict[str, float]:
    flat_value = value.float().reshape(-1)
    return {
        f"{prefix}_min": _as_float(flat_value.min()),
        f"{prefix}_mean": _as_float(flat_value.mean()),
        f"{prefix}_max": _as_float(flat_value.max()),
    }


def _collector_mask(batch: TensorDictBase) -> torch.Tensor | None:
    key = ("collector", "mask")
    if key not in batch.keys(True, True):
        return None
    return batch.get(key).bool()


class LoggingCollectionMetricsHook(TrainerHookBase):
    """Logs BenchMARL-like collection metrics in the ``collection/`` namespace."""

    def __init__(self, group: str = "agents") -> None:
        self.group = group

    def __call__(self, batch: TensorDictBase) -> dict[str, float]:
        keys = set(batch.keys(True, True))
        reward_key = ("next", self.group, "reward")
        if reward_key not in keys:
            return {}

        out: dict[str, float] = {}
        mask = _collector_mask(batch)

        reward = batch.get(reward_key).float()
        if mask is not None:
            reward = reward[mask]
        if reward.numel() > 0:
            out.update(_min_mean_max(f"collection/{self.group}/reward/reward", reward))
            out.update(_min_mean_max("collection/reward/reward", reward))

        done_key = ("next", "done")
        if done_key in keys:
            done = batch.get(done_key).squeeze(-1).bool()
            if mask is not None:
                done = done & mask
            out["collection/done_rate"] = _as_float(done.float().mean())

            episode_key = ("next", self.group, "episode_reward")
            if episode_key in keys and done.any():
                episode_reward = batch.get(episode_key).float().mean(dim=-2).squeeze(-1)
                ended_episode_reward = episode_reward[done]
                if ended_episode_reward.numel() > 0:
                    out.update(
                        _min_mean_max(
                            f"collection/{self.group}/reward/episode_reward",
                            ended_episode_reward,
                        )
                    )
                    out.update(_min_mean_max("collection/reward/episode_reward", ended_episode_reward))

        return out

    def register(self, trainer: Trainer, name: str = "logging_collection_metrics") -> None:
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"group": self.group}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)


class LoggingTrainingMetricsHook(TrainerHookBase):
    """Reduces training tensors and mirrors them under the ``train/`` namespace."""

    def __init__(self, group: str = "agents") -> None:
        self.group = group

    def __call__(self, _sub_batch: TensorDictBase, losses_td: TensorDictBase) -> TensorDictBase:
        for key, value in list(losses_td.items()):
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                value = value.mean()
                losses_td.set(key, value)
            if isinstance(value, torch.Tensor):
                losses_td.set(f"train/{self.group}/{key}", value)
        return losses_td

    def register(self, trainer: Trainer, name: str = "logging_training_metrics") -> None:
        trainer.register_op("process_loss", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"group": self.group}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)


class LoggingCountersHook(TrainerHookBase):
    """Logs counters in the ``counters/`` namespace."""

    def __init__(self, frame_skip: int = 1) -> None:
        self.frame_skip = frame_skip
        self.total_frames = 0
        self.iteration = 0

    def _current_frames(self, batch: TensorDictBase) -> int:
        mask = _collector_mask(batch)
        if mask is not None:
            return int(mask.sum().item() * self.frame_skip)
        return int(batch.numel() * self.frame_skip)

    def __call__(self, batch: TensorDictBase) -> dict[str, int]:
        current_frames = self._current_frames(batch)
        self.total_frames += current_frames
        self.iteration += 1
        return {
            "counters/current_frames": current_frames,
            "counters/total_frames": self.total_frames,
            "counters/iter": self.iteration,
        }

    def register(self, trainer: Trainer, name: str = "logging_counters") -> None:
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "frame_skip": self.frame_skip,
            "total_frames": self.total_frames,
            "iteration": self.iteration,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.frame_skip = int(state_dict.get("frame_skip", self.frame_skip))
        self.total_frames = int(state_dict.get("total_frames", self.total_frames))
        self.iteration = int(state_dict.get("iteration", self.iteration))


class LoggingProgressMetricsHook(TrainerHookBase):
    """Logs a compact progress-bar view for collection and counters metrics."""

    def __init__(self, *, group: str, counters_hook: LoggingCountersHook) -> None:
        self.group = group
        self.counters_hook = counters_hook

    def __call__(self, batch: TensorDictBase) -> dict[str, float | bool]:
        out: dict[str, float | bool] = {
            "counters/total_frames": float(self.counters_hook.total_frames),
            "log_pbar": True,
        }

        reward_key = ("next", self.group, "reward")
        if reward_key in batch.keys(True, True):
            reward = batch.get(reward_key).float()
            mask = _collector_mask(batch)
            if mask is not None:
                reward = reward[mask]
            if reward.numel() > 0:
                out["collection/reward/reward_mean"] = _as_float(reward.mean())
        return out

    def register(self, trainer: Trainer, name: str = "logging_progress_metrics") -> None:
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {"group": self.group}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)


class LoggingEvaluationMetricsHook(TrainerHookBase):
    """Runs periodic evaluation and logs metrics under the ``eval/`` namespace."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        environment,
        group: str,
        interval_frames: int,
        max_steps: int,
        deterministic: bool,
        render: bool,
        video_fps: int,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.group = group
        self.interval_frames = interval_frames
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.render = render
        self.video_fps = video_fps
        self.trainer: Trainer | None = None

    def _render_frame(self) -> np.ndarray | None:
        candidates = [self.environment, getattr(self.environment, "base_env", None)]
        for candidate in candidates:
            raw_env = getattr(candidate, "_env", None)
            if raw_env is None or not hasattr(raw_env, "render"):
                continue
            try:
                frame = raw_env.render(mode="rgb_array")
            except Exception:
                continue
            if isinstance(frame, np.ndarray):
                return frame
        return None

    def _evaluate_once(self, step: int) -> dict[str, float]:
        start = time.perf_counter()
        video_frames: list[np.ndarray] = []

        callback = None
        if self.render:

            def _capture_frame(_env, _td) -> None:
                video_frames.append(self._render_frame())

            callback = _capture_frame

        exploration_type = ExplorationType.DETERMINISTIC if self.deterministic else ExplorationType.RANDOM

        with torch.no_grad(), set_exploration_type(exploration_type):
            rollout = self.environment.rollout(
                max_steps=self.max_steps,
                policy=self.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )

        evaluation_time = time.perf_counter() - start

        reward = rollout.get(("next", self.group, "reward")).float()
        if reward.ndim >= 3:
            episode_return = reward.sum(dim=-3).mean(dim=-2).squeeze(-1)
        else:
            episode_return = reward.reshape(-1)
        episode_return = episode_return.reshape(-1)

        done = rollout.get(("next", "done")).squeeze(-1).bool()
        if done.ndim == 1:
            done = done.unsqueeze(0)
        lengths: list[int] = []
        for trajectory_done in done:
            done_indices = trajectory_done.nonzero(as_tuple=True)[0]
            length = int(done_indices[0].item() + 1) if done_indices.numel() else int(trajectory_done.shape[0])
            lengths.append(length)

        out = {
            "timers/evaluation_time": float(evaluation_time),
            "eval/reward/episode_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        }
        out.update(_min_mean_max(f"eval/{self.group}/reward/episode_reward", episode_return))
        out.update(_min_mean_max("eval/reward/episode_reward", episode_return))

        if video_frames and self.trainer is not None and self.trainer.logger is not None:
            frames = [f for f in video_frames if isinstance(f, np.ndarray)]
            if len(frames) > 1:
                video = torch.as_tensor(
                    np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2)),
                    dtype=torch.uint8,
                ).unsqueeze(0)
                self.trainer.logger.log_video("eval/video", video, step=step, fps=self.video_fps)
            elif self.render:
                warnings.warn(
                    "Evaluation rendering is enabled but no valid frames were captured; eval/video was not logged.",
                    stacklevel=2,
                )

        return out

    def _log_direct(self, metrics: dict[str, float], step: int) -> None:
        if self.trainer is None or self.trainer.logger is None:
            return
        for key, value in metrics.items():
            self.trainer.logger.log_scalar(key, float(value), step=step)

    def run(self, *, step: int) -> dict[str, float]:
        metrics = self._evaluate_once(step=step)
        self._log_direct(metrics, step=step)
        return metrics

    def __call__(self, _batch: TensorDictBase) -> dict[str, float]:
        if self.trainer is None or self.interval_frames <= 0:
            return {}

        frames = int(self.trainer.collected_frames)
        if frames == 0 or frames % self.interval_frames != 0:
            return {}

        return self.run(step=frames)

    def register(self, trainer: Trainer, name: str = "logging_evaluation_metrics") -> None:
        self.trainer = trainer
        trainer.register_op("post_steps_log", self)
        trainer.register_module(name, self)

    def close(self) -> None:
        if hasattr(self.environment, "is_closed") and not self.environment.is_closed:
            self.environment.close()

    def state_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "interval_frames": self.interval_frames,
            "max_steps": self.max_steps,
            "deterministic": self.deterministic,
            "render": self.render,
            "video_fps": self.video_fps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self.interval_frames = int(state_dict.get("interval_frames", self.interval_frames))
        self.max_steps = int(state_dict.get("max_steps", self.max_steps))
        self.deterministic = bool(state_dict.get("deterministic", self.deterministic))
        self.render = bool(state_dict.get("render", self.render))
        self.video_fps = int(state_dict.get("video_fps", self.video_fps))


class LoggingHookSet:
    """Composed logging hooks inspired by BenchMARL defaults."""

    def __init__(
        self,
        *,
        group: str,
        frame_skip: int,
        eval_hook: LoggingEvaluationMetricsHook | None = None,
    ) -> None:
        self.group = group
        self.collection_hook = LoggingCollectionMetricsHook(group=group)
        self.training_hook = LoggingTrainingMetricsHook(group=group)
        self.counters_hook = LoggingCountersHook(frame_skip=frame_skip)
        self.progress_hook = LoggingProgressMetricsHook(group=group, counters_hook=self.counters_hook)
        self.eval_hook = eval_hook

        self._iteration_start: float | None = None
        self._previous_iteration_end: float | None = None
        self._collection_time = 0.0
        self._total_time = 0.0

    def _timers_start(self, batch: TensorDictBase) -> TensorDictBase:
        now = time.perf_counter()
        self._iteration_start = now
        self._collection_time = (
            0.0 if self._previous_iteration_end is None else max(0.0, now - self._previous_iteration_end)
        )
        return batch

    def _timers_end(self, _batch: TensorDictBase) -> dict[str, float]:
        if self._iteration_start is None:
            return {}
        now = time.perf_counter()
        training_time = max(0.0, now - self._iteration_start)
        iteration_time = self._collection_time + training_time
        self._total_time += iteration_time
        self._previous_iteration_end = now
        return {
            "timers/collection_time": float(self._collection_time),
            "timers/training_time": float(training_time),
            "timers/iteration_time": float(iteration_time),
            "timers/total_time": float(self._total_time),
        }

    def register(self, trainer: Trainer) -> None:
        trainer.register_op("batch_process", self._timers_start)
        trainer.register_op("process_loss", self.training_hook)

        trainer.register_op("pre_steps_log", self.counters_hook)
        trainer.register_op("pre_steps_log", self.collection_hook)
        trainer.register_op("pre_steps_log", self.progress_hook)

        trainer.register_op("post_steps_log", self._timers_end)
        if self.eval_hook is not None:
            self.eval_hook.register(trainer)

    def run_pre_eval(self) -> dict[str, float]:
        if self.eval_hook is None:
            return {}
        return self.eval_hook.run(step=0)

    def close(self) -> None:
        if self.eval_hook is not None:
            self.eval_hook.close()
