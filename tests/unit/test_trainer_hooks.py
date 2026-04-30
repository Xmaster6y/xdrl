from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.trainers.trainers import TrainerHookBase

from xdrl.trainer_hooks import (
    LoggingEvaluationHookSet,
    LoggingHookSet,
    PolicyCheckpointHook,
    WandbFinishHook,
    WandbFlushHook,
)
from xdrl.trainer_hooks.logging import (
    LoggingCollectionMetricsHook,
    LoggingCountersHook,
    LoggingEvaluationMetricsHook,
    LoggingTrainingMetricsHook,
)


def _evaluation_hook_for_render(
    environment: object,
    *,
    render_kwargs: dict[str, object] | None = None,
) -> LoggingEvaluationMetricsHook:
    return LoggingEvaluationMetricsHook(
        policy=MagicMock(),
        environment=environment,
        group="agents",
        metric_subgroup="deterministic",
        interval_frames=100,
        max_steps=25,
        deterministic=True,
        render=True,
        render_kwargs=render_kwargs,
        video_fps=20,
        logger=MagicMock(),
    )


def test_policy_checkpoint_hook_saves_periodically(tmp_path):
    policy = torch.nn.Linear(4, 2)
    hook = PolicyCheckpointHook(policy=policy, directory=tmp_path, interval=2, prefix="policy")

    hook()
    assert list(tmp_path.glob("*.pt")) == []

    hook()
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    checkpoint = torch.load(checkpoints[0], map_location="cpu")
    assert checkpoint["step"] == 2
    assert "policy_state_dict" in checkpoint
    assert checkpoint["meta"] == {}
    torch.testing.assert_close(checkpoint["policy_state_dict"]["weight"], policy.state_dict()["weight"])
    torch.testing.assert_close(checkpoint["policy_state_dict"]["bias"], policy.state_dict()["bias"])


def test_policy_checkpoint_hook_inherits_trainer_hook_base():
    assert issubclass(PolicyCheckpointHook, TrainerHookBase)


def test_logging_training_metrics_hook_namespaces_losses():
    losses = TensorDict(
        {
            "loss_objective": torch.tensor([1.0, 3.0]),
            "entropy": torch.tensor([0.2, 0.4]),
        },
        batch_size=[],
    )

    hook = LoggingTrainingMetricsHook(group="agents")
    hook(TensorDict({}, batch_size=[]), losses)

    assert losses.get("loss_objective").shape == torch.Size([])
    assert losses.get("train/agents/loss_objective").item() == pytest.approx(2.0)
    assert losses.get("train/agents/entropy").item() == pytest.approx(0.3)


def test_logging_collection_metrics_hook_emits_collection_namespaces():
    done = torch.tensor([[[False], [True], [False]], [[False], [False], [True]]])
    batch = TensorDict(
        {
            "agents": TensorDict({"observation": torch.zeros(2, 3, 4, 5)}, batch_size=[2, 3, 4]),
            "next": TensorDict(
                {
                    "done": done,
                    "agents": TensorDict(
                        {
                            "reward": torch.ones(2, 3, 4, 1),
                            "episode_reward": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4, 1),
                        },
                        batch_size=[2, 3, 4],
                    ),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = LoggingCollectionMetricsHook(group="agents")
    out = hook(batch)

    assert "collection/agents/reward/reward_mean" in out
    assert "collection/reward/reward_mean" in out
    assert "collection/agents/reward/episode_reward_mean" in out
    assert "collection/reward/episode_reward_mean" in out
    assert out["collection/done_rate"] == pytest.approx(done.float().mean().item())


def test_logging_evaluation_hook_set_creates_configured_variants():
    hook_set = LoggingEvaluationHookSet(
        policy=MagicMock(),
        environment=MagicMock(),
        group="agents",
        interval_frames=100,
        max_steps=25,
        deterministic=True,
        non_deterministic=True,
        render=False,
        video_fps=20,
        logger=MagicMock(),
    )

    assert len(hook_set.hooks) == 2
    assert hook_set.hooks[0].metric_subgroup == "deterministic"
    assert hook_set.hooks[1].metric_subgroup == "non_deterministic"


def test_logging_counters_hook_tracks_total_frames():
    batch = TensorDict(
        {
            "collector": TensorDict({"mask": torch.tensor([[True, False], [True, True]])}, batch_size=[2, 2]),
        },
        batch_size=[2, 2],
    )
    hook = LoggingCountersHook(frame_skip=2)
    out = hook(batch)

    assert out["counters/current_frames"] == 6
    assert out["counters/total_frames"] == 6
    assert out["counters/iter"] == 1


def test_logging_hook_set_run_pre_eval_merges_multiple_eval_hooks():
    eval_hook_set = MagicMock()
    eval_hook_set.run.return_value = {
        "eval/deterministic/reward/episode_len_mean": 10.0,
        "eval/non_deterministic/reward/episode_len_mean": 12.0,
    }

    hook_set = LoggingHookSet(group="agents", frame_skip=1, eval_hook_set=eval_hook_set)

    out = hook_set.run_pre_eval()

    eval_hook_set.run.assert_called_once_with(step=0)
    assert out["eval/deterministic/reward/episode_len_mean"] == 10.0
    assert out["eval/non_deterministic/reward/episode_len_mean"] == 12.0


def test_logging_evaluation_metrics_hook_renders_with_gymnasium_api():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    env = MagicMock()
    env.render.return_value = frame
    hook = _evaluation_hook_for_render(env)

    out = hook._render_frame()

    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 8, 3)
    env.render.assert_called_once_with()


def test_logging_evaluation_metrics_hook_renders_with_explicit_render_kwargs():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    env = MagicMock()
    env.render.return_value = frame
    hook = _evaluation_hook_for_render(env, render_kwargs={"mode": "rgb_array"})

    out = hook._render_frame()

    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 8, 3)
    env.render.assert_called_once_with(mode="rgb_array")


def test_logging_evaluation_metrics_hook_render_fails_when_output_is_none():
    env = MagicMock()
    env.render.return_value = None
    hook = _evaluation_hook_for_render(env)

    with pytest.raises(RuntimeError, match="returned None"):
        hook._render_frame()


def test_logging_evaluation_metrics_hook_render_errors_are_not_swallowed():
    env = MagicMock()
    env.render.side_effect = TypeError("bad kwargs")
    hook = _evaluation_hook_for_render(env, render_kwargs={"mode": "rgb_array"})

    with pytest.raises(TypeError, match="bad kwargs"):
        hook._render_frame()


def test_policy_checkpoint_hook_resolves_policy_from_trainer_path(tmp_path):
    trainer = MagicMock()
    trainer.loss_module.actor_network = torch.nn.Linear(4, 2)
    hook = PolicyCheckpointHook(
        directory=tmp_path,
        interval=1,
        policy_path="loss_module.actor_network",
    )

    hook.register(trainer)
    hook()

    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1
    assert hook.policy is trainer.loss_module.actor_network


def test_disabled_policy_checkpoint_hook_registers_module_only(tmp_path):
    trainer = MagicMock()
    hook = PolicyCheckpointHook(directory=tmp_path, interval=0, policy_path="loss_module.actor_network")

    hook.register(trainer)

    trainer.register_module.assert_called_once()
    trainer.register_op.assert_not_called()


def test_wandb_finish_hook_can_be_disabled():
    trainer = MagicMock()
    hook = WandbFinishHook(enabled=False)

    hook.register(trainer)

    trainer.register_module.assert_called_once()
    trainer.register_op.assert_not_called()


def test_wandb_flush_hook_commits_pending_wandb_rows():
    logger = MagicMock()
    logger._step_registry = {"train/step": 100}
    trainer = MagicMock()
    trainer.logger = logger
    hook = WandbFlushHook()

    hook.register(trainer)
    hook()
    hook()

    logger.experiment.log.assert_called_once_with({}, commit=True)


def test_wandb_flush_hook_ignores_non_wandb_loggers():
    logger = MagicMock()
    logger.experiment.define_metric = None
    logger._step_registry = {"train/step": 100}
    trainer = MagicMock()
    trainer.logger = logger
    hook = WandbFlushHook()

    hook.register(trainer)
    hook()

    logger.experiment.log.assert_not_called()
