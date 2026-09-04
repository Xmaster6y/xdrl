import pytest
import torch
from tdhook.latent import ActivationCaching
from tdhook.workflow import Workflow
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import (
    ActorCriticOperator,
    ActorValueOperator,
    ProbabilisticActor,
    QValueActor,
    QValueModule,
    SafeSequential,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.modules.models.multiagent import QMixer
from torchrl.objectives import ClipPPOLoss, DQNLoss, IQLLoss, LossModule, SACLoss
from torchrl.objectives.multiagent import QMixerLoss

from xdrl import Component, interpret
from xdrl.modules import ModuleInterpretation
from xdrl.objectives import PPOInterpretation

pytestmark = pytest.mark.filterwarnings("ignore:No target network updater has been associated:UserWarning")


class StateActionValue(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((observation, action), dim=-1))


def _actor() -> ProbabilisticActor:
    parameters = TensorDictModule(
        torch.nn.Sequential(torch.nn.Linear(3, 4), NormalParamExtractor()),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    return ProbabilisticActor(
        parameters,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


def _qvalue() -> ValueOperator:
    return ValueOperator(
        StateActionValue(),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )


def test_known_torchrl_modules_expose_their_native_rl_functions() -> None:
    actor = _actor()
    actor_view = interpret(actor)
    assert actor_view.actor.module is actor
    assert actor_view(TensorDict({"observation": torch.zeros(2, 3)}, batch_size=[2]))["action"].shape == (2, 2)
    actor_result = actor_view.run(
        Workflow(ActivationCaching("module.0.module.0", cache_key=("activations", "actor"))),
        TensorDict({"observation": torch.zeros(2, 3)}, batch_size=[2]),
    )
    assert actor_result.data["activations", "actor", "module.0.module.0"].shape == (2, 4)

    qvalue = QValueActor(torch.nn.Linear(3, 2), in_keys=["observation"], action_space="categorical")
    qvalue_view = interpret(qvalue)
    assert isinstance(qvalue_view, ModuleInterpretation)
    assert qvalue_view.policy is qvalue_view.qvalue[0]

    common = TensorDictModule(torch.nn.Linear(3, 4), ["observation"], ["hidden"])
    policy = TensorDictModule(torch.nn.Linear(4, 2), ["hidden"], ["action"])
    value = TensorDictModule(torch.nn.Linear(4, 1), ["hidden"], ["state_action_value"])
    actor_critic = interpret(ActorCriticOperator(common, policy, value))
    assert actor_critic.actor is actor_critic.policy
    assert actor_critic.critic is actor_critic.qvalue[0]
    assert actor_critic.value is None

    value_view = interpret(ValueOperator(torch.nn.Linear(3, 1), in_keys=["observation"]))
    assert value_view.value.name == "value"
    assert value_view.critic is None
    assert value_view.qvalue == ()

    state_action_value = interpret(ValueOperator(StateActionValue(), in_keys=["observation", "action"]))
    assert state_action_value.critic is state_action_value.qvalue[0]
    assert state_action_value.qvalue[0].name == "qvalue"
    assert state_action_value.value is None
    state_action_data = TensorDict(
        {"observation": torch.zeros(2, 3), "action": torch.zeros(2, 2)},
        batch_size=[2],
    )
    assert state_action_value.qvalue[0](state_action_data)["state_action_value"].shape == (2, 1)

    nested_state_action_value = interpret(
        ValueOperator(
            StateActionValue(),
            in_keys=[("agents", "observation"), ("agents", "action")],
        )
    )
    assert nested_state_action_value.critic is nested_state_action_value.qvalue[0]

    actor_value = interpret(
        ActorValueOperator(
            TensorDictModule(torch.nn.Linear(3, 4), ["observation"], ["hidden"]),
            TensorDictModule(torch.nn.Linear(4, 2), ["hidden"], ["action"]),
            TensorDictModule(torch.nn.Linear(4, 1), ["hidden"], ["state_value"]),
        )
    )
    assert actor_value.actor is actor_value.policy
    assert actor_value.critic is actor_value.value


def test_unknown_objectives_fail_instead_of_guessing_network_roles() -> None:
    with pytest.raises(TypeError, match="no objective integration for LossModule"):
        interpret(LossModule())


def test_dqn_exposes_distinct_online_and_target_parameterizations() -> None:
    network = QValueActor(
        torch.nn.Linear(3, 2, bias=False),
        in_keys=["observation"],
        action_space="categorical",
    )
    loss = DQNLoss(network, action_space="categorical", delay_value=True)
    with torch.no_grad():
        loss.value_network_params.apply_(lambda value: value.zero_())
        loss.target_value_network_params.apply_(lambda value: value.fill_(2))

    objective = interpret(loss)
    data = TensorDict({"observation": torch.ones(1, 3)}, batch_size=[1])
    online = objective.qvalue[0](data.clone())["action_value"]
    target = objective.target.qvalue[0](data.clone())["action_value"]
    target_run = objective.target.qvalue[0].run(
        Workflow(ActivationCaching("module.0.module", cache_key=("activations", "target_q"))),
        data.clone(),
    )

    torch.testing.assert_close(online, torch.zeros(1, 2))
    torch.testing.assert_close(target, torch.full((1, 2), 6.0))
    torch.testing.assert_close(target_run.data["action_value"], target)
    torch.testing.assert_close(
        target_run.data["activations", "target_q", "module.0.module"],
        target,
    )


def test_ppo_uses_the_actor_and_critic_owned_by_the_loss() -> None:
    loss = ClipPPOLoss(_actor(), ValueOperator(torch.nn.Linear(3, 1), in_keys=["observation"]))
    objective = interpret(loss)

    assert isinstance(objective, PPOInterpretation)
    assert objective.actor.module is loss.actor_network
    assert objective.critic.module is loss.critic_network
    assert objective.value is objective.critic


def test_sac_and_iql_preserve_qvalue_ensemble_members() -> None:
    sac = interpret(SACLoss(_actor(), _qvalue(), num_qvalue_nets=2))
    iql = interpret(
        IQLLoss(
            _actor(),
            _qvalue(),
            ValueOperator(torch.nn.Linear(3, 1), in_keys=["observation"]),
            num_qvalue_nets=2,
        )
    )

    assert [component.name for component in sac.qvalue] == ["qvalue[0]", "qvalue[1]"]
    assert [component.name for component in sac.target.qvalue] == ["target.qvalue[0]", "target.qvalue[1]"]
    assert len(iql.qvalue) == len(iql.target.qvalue) == 2
    assert iql.actor.module is iql.loss.actor_network
    assert iql.value.module is iql.loss.value_network


def test_sac_exposes_optional_value_and_delayed_actor_components() -> None:
    loss = SACLoss(
        _actor(),
        _qvalue(),
        ValueOperator(torch.nn.Linear(3, 1), in_keys=["observation"]),
        num_qvalue_nets=1,
        delay_actor=True,
        delay_value=True,
    )
    sac = interpret(loss)

    assert len(sac.qvalue) == len(sac.target.qvalue) == 1
    assert sac.value.module is loss.value_network
    assert sac.target.value.params is loss.target_value_network_params
    assert sac.target.actor.params is loss.target_actor_network_params
    data = TensorDict(
        {"observation": torch.zeros(2, 3), "action": torch.zeros(2, 2)},
        batch_size=[2],
    )
    assert sac.qvalue[0](data.clone())["state_action_value"].shape == (2, 1)
    assert sac.target.qvalue[0](data.clone())["state_action_value"].shape == (2, 1)


def test_iql_single_qvalue_member_binds_unbatched_parameters() -> None:
    loss = IQLLoss(
        _actor(),
        _qvalue(),
        ValueOperator(torch.nn.Linear(3, 1), in_keys=["observation"]),
        num_qvalue_nets=1,
    )
    objective = interpret(loss)
    data = TensorDict(
        {"observation": torch.zeros(2, 3), "action": torch.zeros(2, 2)},
        batch_size=[2],
    )

    assert objective.qvalue[0](data.clone())["state_action_value"].shape == (2, 1)
    assert objective.target.qvalue[0](data.clone())["state_action_value"].shape == (2, 1)


def test_qmixer_exposes_local_mixer_and_joint_online_and_target_views() -> None:
    agents = 2
    local_module = TensorDictModule(
        torch.nn.Linear(3, 2),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action_value")],
    )
    selection = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            ("agents", "action"),
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        action_space="categorical",
    )
    local_qvalue = SafeSequential(local_module, selection)
    mixer = TensorDictModule(
        QMixer(state_shape=(4,), mixing_embed_dim=8, n_agents=agents, device="cpu"),
        in_keys=[("agents", "chosen_action_value"), "state"],
        out_keys=["chosen_action_value"],
    )
    loss = QMixerLoss(local_qvalue, mixer, action_space="categorical", delay_value=True)
    with torch.no_grad():
        loss.local_value_network_params.apply_(lambda value: value.zero_())
        loss.mixer_network_params.apply_(lambda value: value.zero_())
        loss.target_local_value_network_params.apply_(lambda value: value.fill_(1))
        loss.target_mixer_network_params.apply_(lambda value: value.fill_(1))
    objective = interpret(loss)
    data = TensorDict(
        {
            "agents": TensorDict({"observation": torch.zeros(5, agents, 3)}, batch_size=[5, agents]),
            "state": torch.zeros(5, 4),
        },
        batch_size=[5],
    )

    assert isinstance(objective.local_qvalue, Component)
    assert isinstance(objective.mixer, Component)
    online = objective.joint_qvalue(data.clone())["chosen_action_value"]
    target = objective.target.joint_qvalue(data.clone())["chosen_action_value"]
    torch.testing.assert_close(online, torch.zeros(5, 1))
    assert not torch.allclose(target, online)
