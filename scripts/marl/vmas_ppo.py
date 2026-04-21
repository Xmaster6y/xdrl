"""Train IPPO or MAPPO on VMAS scenarios with TorchRL PPOTrainer hooks.

Examples:

```bash
uv run -m scripts.marl.vmas_ppo
uv run -m scripts.marl.vmas_ppo algo=mappo
uv run -m scripts.marl.vmas_ppo env.scenario=navigation
```
"""

from __future__ import annotations

import hydra
import torch
from loguru import logger as pylogger
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical

from torchrl.collectors import Collector
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import get_logger
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.trainers import BatchSubSampler, LogScalar

from xdrl.trainer_hooks import MultiAgentGAEHook, ReduceLossTensorsHook


def make_env(cfg: DictConfig) -> TransformedEnv:
    base_env = VmasEnv(
        scenario=cfg.env.scenario,
        num_envs=cfg.env.num_envs,
        continuous_actions=cfg.env.continuous_actions,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario_kwargs,
    )
    return TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )


def make_modules(env: TransformedEnv, cfg: DictConfig) -> tuple[ProbabilisticActor, ValueOperator]:
    group = cfg.model.group
    action_spec = env.full_action_spec_unbatched[env.action_key]
    obs_dim = env.observation_spec[group, "observation"].shape[-1]

    if cfg.env.continuous_actions:
        action_dim = action_spec.shape[-1]
        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=2 * action_dim,
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.share_policy_params,
                device=cfg.train.device,
                depth=cfg.model.depth,
                num_cells=cfg.model.hidden_dim,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        actor_module = TensorDictModule(
            actor_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "loc"), (group, "scale")],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.full_action_spec_unbatched,
            in_keys=[(group, "loc"), (group, "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )
    else:
        n_actions = action_spec.space.n
        actor_module = TensorDictModule(
            MultiAgentMLP(
                n_agent_inputs=obs_dim,
                n_agent_outputs=n_actions,
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.share_policy_params,
                device=cfg.train.device,
                depth=cfg.model.depth,
                num_cells=cfg.model.hidden_dim,
                activation_class=nn.Tanh,
            ),
            in_keys=[(group, "observation")],
            out_keys=[(group, "logits")],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.full_action_spec_unbatched,
            in_keys=[(group, "logits")],
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=(group, "log_prob"),
        )

    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.algo == "mappo",
        share_params=cfg.model.share_critic_params,
        device=cfg.train.device,
        depth=cfg.model.depth,
        num_cells=cfg.model.hidden_dim,
        activation_class=nn.Tanh,
    )
    critic = ValueOperator(
        module=critic_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "state_value")],
    )
    return actor, critic


def resolve_collector_acceleration(
    cfg: DictConfig,
) -> tuple[bool | dict[str, int | str], bool | dict[str, int], str]:
    compile_requested = bool(cfg.collector.get("compile_policy", False))
    cudagraph_requested = bool(cfg.collector.get("cudagraph_policy", False))

    policy_device = torch.device(cfg.train.device)
    cuda_ready = policy_device.type == "cuda" and torch.cuda.is_available()

    if cudagraph_requested and not cuda_ready:
        pylogger.warning(
            "Disabling collector cudagraph_policy because policy device is '{}' and CUDA available is {}.",
            policy_device,
            torch.cuda.is_available(),
        )
        cudagraph_requested = False

    if compile_requested and policy_device.type != "cuda":
        pylogger.warning(
            "collector.compile_policy is enabled on '{}' policy device; speedups are usually smaller than on CUDA.",
            policy_device,
        )

    compile_policy: bool | dict[str, int | str] = False
    if compile_requested:
        compile_policy = {
            "warmup": int(cfg.collector.get("compile_warmup", 1)),
        }
        compile_mode = cfg.collector.get("compile_mode", None)
        if compile_mode is not None:
            compile_policy["mode"] = compile_mode

    cudagraph_policy: bool | dict[str, int] = False
    if cudagraph_requested:
        cudagraph_policy = {
            "warmup": int(cfg.collector.get("cudagraph_warmup", 20)),
        }

    if compile_requested and cudagraph_requested:
        mode = "compile+cudagraph"
    elif cudagraph_requested:
        mode = "cudagraph"
    elif compile_requested:
        mode = "compile"
    else:
        mode = "eager"

    return compile_policy, cudagraph_policy, mode


def make_trainer(cfg: DictConfig, env: TransformedEnv) -> PPOTrainer:
    actor, critic = make_modules(env, cfg)
    compile_policy, cudagraph_policy, collector_mode = resolve_collector_acceleration(cfg)

    pylogger.info(
        "Collector mode={} policy_device={} env_device={} storing_device={}",
        collector_mode,
        cfg.train.device,
        cfg.env.device,
        cfg.train.storing_device,
    )

    collector = Collector(
        env,
        actor,
        policy_device=cfg.train.device,
        env_device=cfg.env.device,
        storing_device=cfg.train.storing_device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        compile_policy=compile_policy,
        cudagraph_policy=cudagraph_policy,
        no_cuda_sync=bool(cfg.collector.get("no_cuda_sync", False)),
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_coef,
        normalize_advantage=False,
    )
    group = cfg.model.group
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
        sample_log_prob=(group, "log_prob"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda)
    loss_module.value_estimator.set_keys(
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    trainer_logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=cfg.logger.log_dir,
        experiment_name=cfg.logger.experiment_name,
        wandb_kwargs={"project": cfg.logger.wandb_project},
    )

    trainer = PPOTrainer(
        collector=collector,
        total_frames=cfg.collector.total_frames,
        frame_skip=cfg.train.frame_skip,
        optim_steps_per_batch=cfg.train.optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=trainer_logger,
        clip_grad_norm=cfg.optim.clip_grad_norm,
        clip_norm=cfg.optim.max_grad_norm,
        progress_bar=cfg.train.progress_bar,
        seed=cfg.seed,
        save_trainer_interval=cfg.train.save_trainer_interval,
        log_interval=cfg.train.log_interval,
        num_epochs=cfg.train.num_epochs,
        replay_buffer=None,
        enable_logging=False,
        add_gae=False,
    )

    trainer.register_op(
        dest="pre_epoch",
        op=MultiAgentGAEHook(
            loss_module=loss_module,
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.lmbda,
            group=group,
        ),
    )
    trainer.register_op(dest="process_loss", op=ReduceLossTensorsHook())
    trainer.register_op(dest="process_optim_batch", op=BatchSubSampler(batch_size=cfg.train.minibatch_size))

    trainer.register_op(
        dest="pre_steps_log",
        op=LogScalar(
            key=("next", group, "reward"),
            logname="reward",
            log_pbar=True,
            include_std=True,
            reduction="mean",
        ),
    )
    trainer.register_op(
        dest="pre_steps_log",
        op=LogScalar(
            key=("next", "done"),
            logname="done_rate",
            log_pbar=False,
            include_std=False,
            reduction="mean",
        ),
    )
    trainer.register_op(
        dest="pre_steps_log",
        op=LogScalar(
            key=("next", group, "episode_reward"),
            logname="episode_reward",
            log_pbar=False,
            include_std=True,
            reduction="mean",
        ),
    )

    return trainer


@hydra.main(config_path="../../configs/marl", config_name="vmas_ppo", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    pylogger.info(
        "Starting {} on VMAS {} with {} envs and {} total frames",
        cfg.algo.upper(),
        cfg.env.scenario,
        cfg.env.num_envs,
        cfg.collector.total_frames,
    )

    env = make_env(cfg)
    trainer = make_trainer(cfg, env)
    try:
        trainer.train()
    finally:
        trainer.collector.shutdown()
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main()
