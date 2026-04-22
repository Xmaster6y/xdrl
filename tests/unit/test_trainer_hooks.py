from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.trainers.trainers import TrainerHookBase

from xdrl.trainer_hooks import (
    ExpandSharedNextKeysHook,
    LoggingEvaluationHookSet,
    LoggingHookSet,
    MultiAgentGAEHook,
    PolicyCheckpointHook,
    ReduceLossTensorsHook,
    WeightedSumRewardHook,
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


def _multi_agent_gae_hook() -> MultiAgentGAEHook:
    loss = MagicMock()
    loss.critic_network = torch.nn.Linear(4, 1)
    loss.critic_network_params = []
    loss.target_critic_network_params = []
    return MultiAgentGAEHook(loss_module=loss, gamma=0.99, lmbda=0.95, group="agents")


def test_multi_agent_gae_hook_runs_gae_when_per_group_next_keys():
    batch = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": torch.zeros(2, 3, 4, 5),
                },
                batch_size=[2, 3, 4],
            ),
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "reward": torch.randn(2, 3, 4, 1),
                            "done": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
                            "terminated": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
                        },
                        batch_size=[2, 3, 4],
                    ),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = _multi_agent_gae_hook()
    hook.gae = MagicMock()
    hook(batch)
    hook.gae.assert_called_once()


def test_multi_agent_gae_hook_raises_when_only_shared_next_keys():
    batch = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": torch.zeros(2, 3, 4, 5),
                },
                batch_size=[2, 3, 4],
            ),
            "next": TensorDict(
                {
                    "reward": torch.randn(2, 3, 1),
                    "done": torch.zeros(2, 3, 1, dtype=torch.bool),
                    "terminated": torch.zeros(2, 3, 1, dtype=torch.bool),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = _multi_agent_gae_hook()
    with pytest.raises(RuntimeError, match="missing"):
        hook(batch)


def test_multi_agent_gae_hook_load_state_updates_gae_keys():
    hook = _multi_agent_gae_hook()
    hook.gae = MagicMock()

    hook.load_state_dict({"group": "players"})

    assert hook.group == "players"
    hook.gae.set_keys.assert_called_once_with(
        reward=("players", "reward"),
        done=("players", "done"),
        terminated=("players", "terminated"),
        advantage=("players", "advantage"),
        value_target=("players", "value_target"),
        value=("players", "state_value"),
    )


def test_expand_shared_next_keys_hook_creates_group_done_and_terminated():
    shared_done = torch.tensor([[[False], [True], [False]], [[True], [False], [True]]])
    shared_terminated = torch.tensor([[[False], [False], [True]], [[False], [True], [False]]])
    batch = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": torch.zeros(2, 3, 4, 5),
                },
                batch_size=[2, 3, 4],
            ),
            "next": TensorDict(
                {
                    "done": shared_done,
                    "terminated": shared_terminated,
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = ExpandSharedNextKeysHook(group="agents")
    hook(batch)

    expanded_done = batch.get(("next", "agents", "done"))
    expanded_terminated = batch.get(("next", "agents", "terminated"))
    assert expanded_done.shape == torch.Size([2, 3, 4, 1])
    assert expanded_terminated.shape == torch.Size([2, 3, 4, 1])
    torch.testing.assert_close(expanded_done, shared_done.unsqueeze(-1).expand(2, 3, 4, 1))
    torch.testing.assert_close(expanded_terminated, shared_terminated.unsqueeze(-1).expand(2, 3, 4, 1))


def test_reduce_loss_tensors_reduces_non_scalars():
    losses = TensorDict(
        {
            "loss_objective": torch.tensor(1.0),
            "explained_variance": torch.ones(2, 3, 1),
        },
        batch_size=[],
    )

    hook = ReduceLossTensorsHook()
    hook(losses, losses)

    assert losses.get("loss_objective").shape == torch.Size([])
    assert losses.get("explained_variance").shape == torch.Size([])
    assert losses.get("explained_variance").item() == pytest.approx(1.0)


def test_marl_hooks_inherit_trainer_hook_base():
    assert issubclass(ExpandSharedNextKeysHook, TrainerHookBase)
    assert issubclass(MultiAgentGAEHook, TrainerHookBase)
    assert issubclass(ReduceLossTensorsHook, TrainerHookBase)


def test_policy_checkpoint_hook_saves_periodically(tmp_path):
    policy = torch.nn.Linear(4, 2)
    hook = PolicyCheckpointHook(policy=policy, directory=tmp_path, interval=2, prefix="policy")
    batch = TensorDict({}, batch_size=[])

    hook(batch)
    assert list(tmp_path.glob("*.pt")) == []

    hook(batch)
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


def test_weighted_sum_reward_default_weights():
    reward = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 1.0]])
    batch = TensorDict(
        {"next": TensorDict({"reward": reward}, batch_size=[2])},
        batch_size=[2],
    )
    hook = WeightedSumRewardHook(weights=None, preserve_input_key=None)
    hook(batch)

    scalar = batch.get(("next", "reward"))
    assert scalar.shape == torch.Size([2, 1])
    assert scalar.squeeze(-1).tolist() == pytest.approx([6.0, 2.0])


def test_weighted_sum_reward_with_custom_weights():
    reward = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    batch = TensorDict(
        {"next": TensorDict({"reward": reward}, batch_size=[1, 2])},
        batch_size=[1, 2],
    )
    hook = WeightedSumRewardHook(weights=[0.2, 0.8], preserve_input_key=None)
    hook(batch)

    scalar = batch.get(("next", "reward"))
    assert scalar.shape == torch.Size([1, 2, 1])
    torch.testing.assert_close(scalar.squeeze(-1), torch.tensor([[1.8, 3.8]]))


def test_weighted_sum_reward_hook_overwrites_and_preserves_vector_reward():
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "reward": torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
                },
                batch_size=[2],
            )
        },
        batch_size=[2],
    )
    hook = WeightedSumRewardHook(weights=[0.5, 0.5])

    hook(batch)

    assert batch.get(("next", "reward")).shape == torch.Size([2, 1])
    assert batch.get(("next", "reward")).squeeze(-1).tolist() == pytest.approx([1.5, 3.0])
    assert batch.get(("next", "reward_vector")).shape == torch.Size([2, 2])


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


def test_logging_collection_metrics_hook_supports_custom_reward_key():
    done = torch.tensor([[[False], [True], [False]], [[False], [False], [True]]])
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": done,
                    "reward": torch.ones(2, 3, 1),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = LoggingCollectionMetricsHook(
        group="agent",
        reward_key=("next", "reward"),
        done_key=("next", "done"),
        episode_reward_key=None,
    )
    out = hook(batch)

    assert "collection/agent/reward/reward_mean" in out
    assert "collection/reward/reward_mean" in out
    assert "collection/done_rate" in out


def test_logging_collection_metrics_hook_scalarizes_vector_episode_reward():
    done = torch.tensor([[[False], [True], [False]], [[False], [False], [True]]])
    batch = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": done,
                    "reward": torch.ones(2, 3, 1),
                    "episode_reward": torch.tensor(
                        [
                            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
                            [[4.0, 8.0], [5.0, 10.0], [6.0, 12.0]],
                        ]
                    ),
                },
                batch_size=[2, 3],
            ),
        },
        batch_size=[2, 3],
    )

    hook = LoggingCollectionMetricsHook(
        group="agent",
        reward_key=("next", "reward"),
        done_key=("next", "done"),
        episode_reward_key=("next", "episode_reward"),
        episode_reward_weights=[0.2, 0.8],
    )
    out = hook(batch)

    assert out["collection/agent/reward/episode_reward_min"] == pytest.approx(3.6)
    assert out["collection/agent/reward/episode_reward_mean"] == pytest.approx(7.2)
    assert out["collection/agent/reward/episode_reward_max"] == pytest.approx(10.8)


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


def test_logging_hook_set_type_exists():
    assert LoggingHookSet is not None


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


def test_logging_hook_set_registers_and_closes_multiple_eval_hooks():
    eval_hook_set = MagicMock()
    trainer = MagicMock()

    hook_set = LoggingHookSet(group="agents", frame_skip=1, eval_hook_set=eval_hook_set)
    hook_set.register(trainer)
    hook_set.close()

    eval_hook_set.register.assert_called_once_with(trainer)
    eval_hook_set.close.assert_called_once()


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
