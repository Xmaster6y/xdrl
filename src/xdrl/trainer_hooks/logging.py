from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs import ExplorationType
from torchrl.envs.utils import set_exploration_type
from torchrl.trainers.trainers import Trainer, TrainerHookBase

from xdrl.trainer_hooks._utils import _as_float


def _min_mean_max(prefix: str, value: torch.Tensor) -> dict[str, float]:
    flat_value = value.float().reshape(-1)
    return {
        f"{prefix}_min": _as_float(flat_value.min()),
        f"{prefix}_mean": _as_float(flat_value.mean()),
        f"{prefix}_max": _as_float(flat_value.max()),
    }


def _summarize_metric(prefix: str, value: torch.Tensor, *, reduce_stats: bool | None = None) -> dict[str, float]:
    if reduce_stats is None:
        reduce_stats = value.numel() > 1
    if reduce_stats:
        return _min_mean_max(prefix, value)
    return {prefix: _as_float(value.reshape(-1)[0])}


def _collector_mask(batch: TensorDictBase) -> torch.Tensor | None:
    key = ("collector", "mask")
    if key not in batch.keys(True, True):
        return None
    return batch.get(key).bool()


def _scalarize_last_dim(value: torch.Tensor, weights: tuple[float, ...] | None) -> torch.Tensor:
    if value.ndim == 0:
        return value

    if value.shape[-1] == 1:
        return value.squeeze(-1)

    if weights is None:
        return value.sum(dim=-1)

    if len(weights) != value.shape[-1]:
        msg = (
            "episode_reward objective dimension does not match scalarization weights: "
            f"got {value.shape[-1]} objectives and {len(weights)} weights"
        )
        raise ValueError(msg)

    weights_tensor = torch.as_tensor(weights, device=value.device, dtype=value.dtype)
    return (value * weights_tensor).sum(dim=-1)


def _prepare_episode_reward(
    value: torch.Tensor,
    done: torch.Tensor,
    *,
    weights: tuple[float, ...] | None,
) -> torch.Tensor:
    if value.shape[: done.ndim] != done.shape:
        msg = (
            "episode_reward shape must share done-prefix dimensions: "
            f"episode_reward={tuple(value.shape)} done={tuple(done.shape)}"
        )
        raise ValueError(msg)

    if value.ndim == done.ndim:
        return value

    reduce_dims = tuple(range(done.ndim, value.ndim - 1))
    if reduce_dims:
        value = value.mean(dim=reduce_dims)
    return _scalarize_last_dim(value, weights)


class LoggingCollectionMetricsHook(TrainerHookBase):
    """Logs BenchMARL-like collection metrics in the ``collection/`` namespace."""

    def __init__(
        self,
        group: str = "agents",
        reward_key: tuple[str, ...] | None = None,
        done_key: tuple[str, ...] = ("next", "done"),
        episode_reward_key: tuple[str, ...] | None = None,
        episode_reward_weights: Sequence[float] | None = None,
        reduce_stats: bool | None = None,
    ) -> None:
        self.group = group
        self.reward_key = reward_key if reward_key is not None else ("next", group, "reward")
        self.done_key = done_key
        self.episode_reward_key = (
            episode_reward_key if episode_reward_key is not None else ("next", group, "episode_reward")
        )
        self.episode_reward_weights = (
            tuple(float(weight) for weight in episode_reward_weights) if episode_reward_weights is not None else None
        )
        self.reduce_stats = reduce_stats

    def __call__(self, batch: TensorDictBase) -> dict[str, float]:
        keys = set(batch.keys(True, True))
        if self.reward_key not in keys:
            return {}

        out: dict[str, float] = {}
        mask = _collector_mask(batch)

        reward = batch.get(self.reward_key).float()
        if mask is not None:
            reward = reward[mask]
        if reward.numel() > 0:
            out.update(
                _summarize_metric(
                    f"collection/{self.group}/reward/reward",
                    reward,
                    reduce_stats=self.reduce_stats,
                )
            )
            out.update(_summarize_metric("collection/reward/reward", reward, reduce_stats=self.reduce_stats))

        if self.done_key in keys:
            done = batch.get(self.done_key).squeeze(-1).bool()
            if mask is not None:
                done = done & mask
            out["collection/done_rate"] = _as_float(done.float().mean())

            if self.episode_reward_key is not None and self.episode_reward_key in keys and done.any():
                episode_reward = _prepare_episode_reward(
                    batch.get(self.episode_reward_key).float(),
                    done,
                    weights=self.episode_reward_weights,
                )
                ended_episode_reward = episode_reward[done]
                if ended_episode_reward.numel() > 0:
                    out.update(
                        _summarize_metric(
                            f"collection/{self.group}/reward/episode_reward",
                            ended_episode_reward,
                            reduce_stats=self.reduce_stats,
                        )
                    )
                    out.update(
                        _summarize_metric(
                            "collection/reward/episode_reward",
                            ended_episode_reward,
                            reduce_stats=self.reduce_stats,
                        )
                    )

        return out

    def register(self, trainer: Trainer, name: str = "logging_collection_metrics") -> None:
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "reward_key": self.reward_key,
            "done_key": self.done_key,
            "episode_reward_key": self.episode_reward_key,
            "episode_reward_weights": self.episode_reward_weights,
            "reduce_stats": self.reduce_stats,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self.reward_key = tuple(state_dict.get("reward_key", self.reward_key))
        self.done_key = tuple(state_dict.get("done_key", self.done_key))
        episode_reward_key = state_dict.get("episode_reward_key", self.episode_reward_key)
        self.episode_reward_key = None if episode_reward_key is None else tuple(episode_reward_key)
        episode_reward_weights = state_dict.get("episode_reward_weights", self.episode_reward_weights)
        self.episode_reward_weights = (
            None if episode_reward_weights is None else tuple(float(weight) for weight in episode_reward_weights)
        )
        reduce_stats = state_dict.get("reduce_stats", self.reduce_stats)
        self.reduce_stats = None if reduce_stats is None else bool(reduce_stats)


class LoggingTrainingMetricsHook(TrainerHookBase):
    """Logs reduced optimization metrics under the ``train/`` namespace."""

    def __init__(self, group: str = "agents") -> None:
        self.group = group

    def __call__(self, _optim_steps: int, average_losses: TensorDictBase | None) -> dict[str, float]:
        if average_losses is None:
            return {}

        out: dict[str, float] = {}
        for key, value in list(average_losses.items()):
            if not isinstance(value, torch.Tensor):
                continue
            scalar = value.mean() if value.numel() > 1 else value.reshape(())
            average_losses.set(key, scalar)
            namespaced_key = f"train/{self.group}/{key}"
            average_losses.set(namespaced_key, scalar)
            out[namespaced_key] = _as_float(scalar)
        return out

    def register(self, trainer: Trainer, name: str = "logging_training_metrics") -> None:
        trainer.register_op("post_optim_complete_log", self)
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

    def __init__(
        self, *, group: str, counters_hook: LoggingCountersHook, reward_key: tuple[str, ...] | None = None
    ) -> None:
        self.group = group
        self.counters_hook = counters_hook
        self.reward_key = reward_key if reward_key is not None else ("next", group, "reward")

    def __call__(self, batch: TensorDictBase) -> dict[str, float | bool]:
        out: dict[str, float | bool] = {
            "counters/total_frames": float(self.counters_hook.total_frames),
            "log_pbar": True,
        }

        if self.reward_key in batch.keys(True, True):
            reward = batch.get(self.reward_key).float()
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
        return {"group": self.group, "reward_key": self.reward_key}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self.reward_key = tuple(state_dict.get("reward_key", self.reward_key))


class LoggingEvaluationMetricsHook(TrainerHookBase):
    """Runs periodic evaluation and logs metrics under ``eval/<subgroup>/``."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        environment,
        group: str,
        metric_subgroup: str,
        interval_frames: int,
        max_steps: int,
        deterministic: bool,
        render: bool,
        video_fps: int,
        render_kwargs: dict[str, Any] | None = None,
        reward_key: tuple[str, ...] | None = None,
        reduce_stats: bool | None = None,
        logger: Any | None = None,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.group = group
        self.reward_key = reward_key if reward_key is not None else ("next", group, "reward")
        self.reduce_stats = reduce_stats
        self.metric_subgroup = metric_subgroup
        self.interval_frames = interval_frames
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.render = render
        self.render_kwargs = dict(render_kwargs) if render_kwargs is not None else None
        self.video_fps = video_fps
        self.logger = logger
        self.trainer: Trainer | None = None

    @staticmethod
    def _to_frame_array(frame: Any) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if isinstance(frame, np.ndarray):
            return frame
        msg = f"Expected render output to be a numpy array or torch tensor, got {type(frame).__name__}."
        raise RuntimeError(msg)

    def _extract_render_frame(self, output: Any) -> np.ndarray:
        if output is None:
            msg = "Render returned None while evaluation rendering is enabled."
            raise RuntimeError(msg)
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, (np.ndarray, torch.Tensor)):
                    return self._to_frame_array(item)
            msg = "Render returned a sequence but contained no numpy array or torch tensor frame."
            raise RuntimeError(msg)
        return self._to_frame_array(output)

    def _renderable_candidates(self) -> list[Any]:
        candidates = [
            self.environment,
            getattr(self.environment, "base_env", None),
            getattr(self.environment, "_env", None),
        ]
        envs = getattr(self.environment, "envs", None)
        if isinstance(envs, (list, tuple)):
            candidates.extend(envs)
        return [candidate for candidate in candidates if candidate is not None and hasattr(candidate, "render")]

    def _render_frame(self) -> np.ndarray:
        candidates = self._renderable_candidates()
        if not candidates:
            msg = "Evaluation rendering is enabled but no renderable environment candidate was found."
            raise RuntimeError(msg)

        candidate = candidates[0]
        kwargs = self.render_kwargs or {}
        output = candidate.render(**kwargs)
        return self._extract_render_frame(output)

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

        reward = rollout.get(self.reward_key).float()
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
            f"timers/eval/{self.metric_subgroup}/evaluation_time": float(evaluation_time),
            f"eval/{self.metric_subgroup}/reward/episode_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        }
        out.update(
            _summarize_metric(
                f"eval/{self.metric_subgroup}/{self.group}/reward/episode_reward",
                episode_return,
                reduce_stats=self.reduce_stats,
            )
        )
        out.update(
            _summarize_metric(
                f"eval/{self.metric_subgroup}/reward/episode_reward",
                episode_return,
                reduce_stats=self.reduce_stats,
            )
        )

        target_logger = self.trainer.logger if self.trainer is not None else self.logger
        if self.render and not video_frames:
            msg = "Evaluation rendering is enabled but no frames were captured during rollout."
            raise RuntimeError(msg)

        if video_frames and target_logger is not None:
            if len(video_frames) >= 1:
                video = torch.as_tensor(
                    np.transpose(np.stack(video_frames, axis=0), (0, 3, 1, 2)),
                    dtype=torch.uint8,
                ).unsqueeze(0)
                target_logger.log_video(
                    f"eval/{self.metric_subgroup}/video",
                    video,
                    step=step,
                    fps=self.video_fps,
                )

        return out

    def _log_direct(self, metrics: dict[str, float], step: int) -> None:
        target_logger = self.trainer.logger if self.trainer is not None else self.logger
        if target_logger is None:
            return
        for key, value in metrics.items():
            target_logger.log_scalar(key, float(value), step=step)

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
            "reward_key": self.reward_key,
            "reduce_stats": self.reduce_stats,
            "metric_subgroup": self.metric_subgroup,
            "interval_frames": self.interval_frames,
            "max_steps": self.max_steps,
            "deterministic": self.deterministic,
            "render": self.render,
            "render_kwargs": self.render_kwargs,
            "video_fps": self.video_fps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.group = state_dict.get("group", self.group)
        self.reward_key = tuple(state_dict.get("reward_key", self.reward_key))
        reduce_stats = state_dict.get("reduce_stats", self.reduce_stats)
        self.reduce_stats = None if reduce_stats is None else bool(reduce_stats)
        self.metric_subgroup = state_dict.get("metric_subgroup", self.metric_subgroup)
        self.interval_frames = int(state_dict.get("interval_frames", self.interval_frames))
        self.max_steps = int(state_dict.get("max_steps", self.max_steps))
        self.deterministic = bool(state_dict.get("deterministic", self.deterministic))
        self.render = bool(state_dict.get("render", self.render))
        render_kwargs = state_dict.get("render_kwargs", self.render_kwargs)
        self.render_kwargs = None if render_kwargs is None else dict(render_kwargs)
        self.video_fps = int(state_dict.get("video_fps", self.video_fps))


class LoggingEvaluationHookSet:
    """Composed deterministic/non-deterministic evaluation hooks."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        environment,
        group: str,
        interval_frames: int,
        max_steps: int,
        deterministic: bool,
        non_deterministic: bool,
        render: bool,
        video_fps: int,
        render_kwargs: dict[str, Any] | None = None,
        reward_key: tuple[str, ...] | None = None,
        reduce_stats: bool | None = None,
        logger: Any | None = None,
    ) -> None:
        self.hooks: list[LoggingEvaluationMetricsHook] = []
        shared_kwargs = {
            "policy": policy,
            "environment": environment,
            "group": group,
            "reward_key": reward_key,
            "reduce_stats": reduce_stats,
            "interval_frames": interval_frames,
            "max_steps": max_steps,
            "render": render,
            "render_kwargs": render_kwargs,
            "video_fps": video_fps,
            "logger": logger,
        }

        if deterministic:
            self.hooks.append(
                LoggingEvaluationMetricsHook(
                    metric_subgroup="deterministic",
                    deterministic=True,
                    **shared_kwargs,
                )
            )

        if non_deterministic:
            self.hooks.append(
                LoggingEvaluationMetricsHook(
                    metric_subgroup="non_deterministic",
                    deterministic=False,
                    **shared_kwargs,
                )
            )

    def register(self, trainer: Trainer, name: str = "logging_evaluation_metrics") -> None:
        for idx, hook in enumerate(self.hooks):
            hook.register(trainer, name=f"{name}_{idx}")

    def run(self, *, step: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for hook in self.hooks:
            out.update(hook.run(step=step))
        return out

    def close(self) -> None:
        for hook in self.hooks:
            hook.close()


class LoggingHookSet:
    """Composed logging hooks inspired by BenchMARL defaults."""

    def __init__(
        self,
        *,
        group: str,
        frame_skip: int,
        reward_key: tuple[str, ...] | None = None,
        done_key: tuple[str, ...] = ("next", "done"),
        episode_reward_key: tuple[str, ...] | None = None,
        episode_reward_weights: Sequence[float] | None = None,
        reduce_stats: bool | None = None,
        eval_hook_set: LoggingEvaluationHookSet | None = None,
    ) -> None:
        self.group = group
        self.collection_hook = LoggingCollectionMetricsHook(
            group=group,
            reward_key=reward_key,
            done_key=done_key,
            episode_reward_key=episode_reward_key,
            episode_reward_weights=episode_reward_weights,
            reduce_stats=reduce_stats,
        )
        self.training_hook = LoggingTrainingMetricsHook(group=group)
        self.counters_hook = LoggingCountersHook(frame_skip=frame_skip)
        self.progress_hook = LoggingProgressMetricsHook(
            group=group,
            counters_hook=self.counters_hook,
            reward_key=reward_key,
        )
        self.eval_hook_set = eval_hook_set

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
        trainer.register_op("post_optim_complete_log", self.training_hook)

        trainer.register_op("pre_steps_log", self.counters_hook)
        trainer.register_op("pre_steps_log", self.collection_hook)
        trainer.register_op("pre_steps_log", self.progress_hook)

        trainer.register_op("post_steps_log", self._timers_end)
        if self.eval_hook_set is not None:
            self.eval_hook_set.register(trainer)

    def run_pre_eval(self) -> dict[str, float]:
        if self.eval_hook_set is None:
            return {}
        return self.eval_hook_set.run(step=0)

    def close(self) -> None:
        if self.eval_hook_set is not None:
            self.eval_hook_set.close()
