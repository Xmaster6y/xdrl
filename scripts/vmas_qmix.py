"""Train QMIX on VMAS scenarios with a BenchMARL-inspired setup.

Examples:

```bash
uv run -m scripts.vmas_qmix
uv run -m scripts.vmas_qmix env.scenario=navigation
uv run -m scripts.vmas_qmix collector.total_frames=300000
```
"""

from __future__ import annotations

import time

import hydra
import numpy as np
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import ExplorationType, RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer
from torchrl.objectives import HardUpdate, SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss
from torchrl.record.loggers import get_logger


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _min_mean_max(prefix: str, value: torch.Tensor) -> dict[str, float]:
    flat_value = value.float().reshape(-1)
    return {
        f"{prefix}_min": _as_float(flat_value.min()),
        f"{prefix}_mean": _as_float(flat_value.mean()),
        f"{prefix}_max": _as_float(flat_value.max()),
    }


def make_env(cfg: DictConfig, *, num_envs: int) -> TransformedEnv:
    base_env = VmasEnv(
        scenario=cfg.env.scenario,
        num_envs=num_envs,
        continuous_actions=False,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario_kwargs,
    )
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


def make_modules(
    env: TransformedEnv,
    cfg: DictConfig,
) -> tuple[SafeSequential, TensorDictSequential, TensorDictModule]:
    group = cfg.model.group
    obs_dim = env.observation_spec[group, "observation"].shape[-1]
    n_actions = int(env.full_action_spec_unbatched[env.action_key].space.n)

    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=n_actions,
        n_agents=env.n_agents,
        centralised=False,
        share_params=cfg.model.share_policy_params,
        device=cfg.train.device,
        depth=cfg.model.depth,
        num_cells=cfg.model.hidden_dim,
        activation_class=nn.Tanh,
    )
    module = TensorDictModule(
        net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "action_value")],
    )
    value_module = QValueModule(
        action_value_key=(group, "action_value"),
        out_keys=[
            env.action_key,
            (group, "action_value"),
            (group, "chosen_action_value"),
        ],
        spec=env.full_action_spec_unbatched,
    )
    qnet = SafeSequential(module, value_module)

    qnet_explore = TensorDictSequential(
        qnet,
        EGreedyModule(
            eps_init=cfg.exploration.eps_init,
            eps_end=cfg.exploration.eps_end,
            annealing_num_steps=cfg.exploration.anneal_frames,
            action_key=env.action_key,
            spec=env.full_action_spec_unbatched,
        ),
    )

    mixer = TensorDictModule(
        module=QMixer(
            state_shape=env.observation_spec_unbatched[group, "observation"].shape,
            mixing_embed_dim=cfg.model.mixing_embed_dim,
            n_agents=env.n_agents,
            device=cfg.train.device,
        ),
        in_keys=[(group, "chosen_action_value"), (group, "observation")],
        out_keys=["chosen_action_value"],
    )
    return qnet, qnet_explore, mixer


def _render_frame(environment: TransformedEnv) -> np.ndarray | None:
    candidates = [environment, getattr(environment, "base_env", None)]
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


def evaluate(
    *,
    qnet: SafeSequential,
    environment: TransformedEnv,
    cfg: DictConfig,
    logger,
    step: int,
) -> dict[str, float]:
    start = time.perf_counter()
    video_frames: list[np.ndarray] = []

    callback = None
    if cfg.eval.render:

        def _capture_frame(_env, _td) -> None:
            frame = _render_frame(environment)
            if frame is not None:
                video_frames.append(frame)

        callback = _capture_frame

    exploration_type = ExplorationType.DETERMINISTIC if cfg.eval.deterministic else ExplorationType.RANDOM
    with torch.no_grad(), set_exploration_type(exploration_type):
        rollout = environment.rollout(
            max_steps=cfg.eval.max_steps,
            policy=qnet,
            callback=callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

    eval_time = time.perf_counter() - start
    reward = rollout.get(("next", cfg.model.group, "reward")).float()
    episode_return = reward.sum(dim=-3).mean(dim=-2).squeeze(-1).reshape(-1)

    done = rollout.get(("next", "done")).squeeze(-1).bool()
    if done.ndim == 1:
        done = done.unsqueeze(0)
    lengths: list[int] = []
    for trajectory_done in done:
        done_indices = trajectory_done.nonzero(as_tuple=True)[0]
        length = int(done_indices[0].item() + 1) if done_indices.numel() else int(trajectory_done.shape[0])
        lengths.append(length)

    metrics = {
        "timers/evaluation_time": float(eval_time),
        "eval/reward/episode_len_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    metrics.update(_min_mean_max("eval/reward/episode_reward", episode_return))
    metrics.update(_min_mean_max(f"eval/{cfg.model.group}/reward/episode_reward", episode_return))

    for key, value in metrics.items():
        logger.log_scalar(key, float(value), step=step)

    if cfg.eval.render and len(video_frames) > 1:
        video = torch.as_tensor(
            np.transpose(np.stack(video_frames, axis=0), (0, 3, 1, 2)),
            dtype=torch.uint8,
        ).unsqueeze(0)
        logger.log_video("eval/video", video, step=step, fps=cfg.eval.video_fps)

    return metrics


def _prepare_batch_for_qmix(batch: TensorDictBase, group: str) -> TensorDictBase:
    reward = batch.get(("next", group, "reward")).mean(dim=-2)
    batch.set(("next", "reward"), reward)
    return batch


@hydra.main(config_path="../configs", config_name="vmas_qmix", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.env.continuous_actions:
        msg = "QMIX only supports discrete actions. Set env.continuous_actions=false."
        raise ValueError(msg)

    torch.manual_seed(cfg.seed)

    pylogger.info(
        "Starting QMIX on VMAS {} with {} envs and {} total frames",
        cfg.env.scenario,
        cfg.env.num_envs,
        cfg.collector.total_frames,
    )

    env = make_env(cfg, num_envs=cfg.env.num_envs)
    eval_env = make_env(cfg, num_envs=cfg.eval.episodes)

    qnet, qnet_explore, mixer = make_modules(env, cfg)

    collector = Collector(
        env,
        qnet_explore,
        policy_device=cfg.train.device,
        env_device=cfg.env.device,
        storing_device=cfg.train.storing_device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        no_cuda_sync=cfg.collector.no_cuda_sync,
        init_random_frames=cfg.buffer.init_random_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.storing_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = QMixerLoss(
        qnet,
        mixer,
        delay_value=cfg.loss.delay_value,
        loss_function=cfg.loss.loss_function,
    )
    loss_module.set_keys(
        action_value=(cfg.model.group, "action_value"),
        local_value=(cfg.model.group, "chosen_action_value"),
        global_value="chosen_action_value",
        action=env.action_key,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)

    if cfg.loss.soft_target_update:
        target_net_updater = SoftUpdate(loss_module, tau=cfg.loss.tau)
    else:
        target_net_updater = HardUpdate(
            loss_module,
            value_network_update_interval=cfg.loss.hard_target_update_interval,
        )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    exp_logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=cfg.logger.experiment_name,
        wandb_kwargs={"project": cfg.logger.wandb_project},
    )
    exp_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    total_frames = 0
    total_time = 0.0
    sampling_start = time.perf_counter()

    try:
        if cfg.eval.enabled and cfg.eval.pre_eval:
            pylogger.info("Running pre-training evaluation")
            evaluate(qnet=qnet, environment=eval_env, cfg=cfg, logger=exp_logger, step=0)

        for iteration, batch in enumerate(collector, start=1):
            sampling_time = time.perf_counter() - sampling_start

            current_frames = int(batch.numel() * cfg.train.frame_skip)
            total_frames += current_frames

            batch = _prepare_batch_for_qmix(batch, cfg.model.group)
            replay_buffer.extend(batch.reshape(-1))

            training_start = time.perf_counter()
            loss_values: list[float] = []
            grad_norm_values: list[float] = []

            if len(replay_buffer) >= cfg.train.minibatch_size:
                for _ in range(cfg.train.optim_steps_per_batch):
                    sub_batch = replay_buffer.sample()
                    loss_td = loss_module(sub_batch)
                    loss = loss_td["loss"]

                    optimizer.zero_grad()
                    loss.backward()

                    grad_norm: float | None = None
                    if cfg.optim.clip_grad_norm:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            loss_module.parameters(),
                            cfg.optim.max_grad_norm,
                        )
                        grad_norm = float(grad_norm_tensor.item())

                    optimizer.step()
                    target_net_updater.step()

                    loss_values.append(float(loss.item()))
                    if grad_norm is not None:
                        grad_norm_values.append(grad_norm)

            training_time = time.perf_counter() - training_start
            iteration_time = sampling_time + training_time
            total_time += iteration_time

            qnet_explore[1].step(frames=current_frames)
            collector.update_policy_weights_()

            metrics: dict[str, float] = {
                "counters/iter": float(iteration),
                "counters/current_frames": float(current_frames),
                "counters/total_frames": float(total_frames),
                "timers/collection_time": float(sampling_time),
                "timers/training_time": float(training_time),
                "timers/iteration_time": float(iteration_time),
                "timers/total_time": float(total_time),
                "collection/done_rate": float(batch.get(("next", "done")).float().mean().item()),
                "train/epsilon": float(qnet_explore[1].eps.item()),
            }

            reward = batch.get(("next", cfg.model.group, "reward")).float()
            metrics.update(_min_mean_max("collection/reward/reward", reward))
            metrics.update(_min_mean_max(f"collection/{cfg.model.group}/reward/reward", reward))

            if loss_values:
                metrics["train/loss"] = float(np.mean(loss_values))
            if grad_norm_values:
                metrics["train/grad_norm"] = float(np.mean(grad_norm_values))

            for key, value in metrics.items():
                exp_logger.log_scalar(key, value, step=total_frames)

            if total_frames % cfg.train.log_interval == 0:
                pylogger.info(
                    "iter={} frames={} replay={} epsilon={:.3f} loss={}",
                    iteration,
                    total_frames,
                    len(replay_buffer),
                    metrics["train/epsilon"],
                    "n/a" if "train/loss" not in metrics else f"{metrics['train/loss']:.4f}",
                )

            if (
                cfg.eval.enabled
                and cfg.eval.interval_frames > 0
                and total_frames % cfg.eval.interval_frames == 0
            ):
                evaluate(qnet=qnet, environment=eval_env, cfg=cfg, logger=exp_logger, step=total_frames)

            sampling_start = time.perf_counter()

        pylogger.info("Training finished after {} frames", total_frames)
    finally:
        collector.shutdown()
        if not env.is_closed:
            env.close()
        if not eval_env.is_closed:
            eval_env.close()


if __name__ == "__main__":
    main()
