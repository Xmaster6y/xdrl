from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from torchrl.trainers.algorithms.configs import *  # noqa: F401,F403

from xdrl.configs import register_configs


def test_register_configs_supports_vmas_configs() -> None:
    register_configs()
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg_vmas_ppo = compose(config_name="vmas_ppo")
        cfg_vmas_qmix = compose(config_name="vmas_qmix")

    OmegaConf.resolve(cfg_vmas_ppo)
    OmegaConf.resolve(cfg_vmas_qmix)

    assert cfg_vmas_ppo.trainer._target_ == "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"
    assert cfg_vmas_qmix.trainer._target_ == "torchrl.trainers.algorithms.configs.trainers._make_dqn_trainer"
    assert [hook._target_ for hook in cfg_vmas_ppo.trainer.hooks] == [
        "xdrl.configs.hooks._make_logging_hook_set",
        "xdrl.configs.hooks._make_log_validation_reward_hook",
        "xdrl.configs.hooks._make_policy_checkpoint_hook",
        "xdrl.trainer_hooks.logging.WandbFinishHook",
    ]
    assert [hook._target_ for hook in cfg_vmas_qmix.trainer.hooks] == [
        "xdrl.configs.hooks._make_logging_hook_set",
        "xdrl.configs.hooks._make_log_validation_reward_hook",
        "xdrl.trainer_hooks.logging.WandbFinishHook",
    ]
    assert cfg_vmas_qmix.trainer.mixing_strategy == "qmix"
    assert list(cfg_vmas_qmix.trainer.reward_key) == ["agents", "reward"]
    assert list(cfg_vmas_qmix.trainer.episode_reward_key) == ["agents", "episode_reward"]
    assert cfg_vmas_qmix.trainer.aggregated_reward_key == "reward"
    assert cfg_vmas_qmix.trainer.aggregated_episode_reward_key == "episode_reward"
    assert cfg_vmas_qmix.mixer_loss.reward_key == "reward"


def test_register_configs_supports_gymnasium_configs() -> None:
    register_configs()
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg_gym_dqn = compose(config_name="gymnasium_dqn")
        cfg_mogym_ppo = compose(config_name="mogymnasium_ppo")

    OmegaConf.resolve(cfg_gym_dqn)
    OmegaConf.resolve(cfg_mogym_ppo)

    assert cfg_gym_dqn.trainer._target_ == "torchrl.trainers.algorithms.configs.trainers._make_dqn_trainer"
    assert cfg_mogym_ppo.trainer._target_ == "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"
    assert [hook._target_ for hook in cfg_gym_dqn.trainer.hooks] == [
        "xdrl.configs.hooks._make_logging_hook_set",
        "xdrl.configs.hooks._make_log_validation_reward_hook",
        "xdrl.trainer_hooks.logging.WandbFinishHook",
    ]
    assert [hook._target_ for hook in cfg_mogym_ppo.trainer.hooks] == [
        "xdrl.configs.hooks.GAEHook",
        "xdrl.configs.hooks._make_logging_hook_set",
        "xdrl.configs.hooks._make_log_validation_reward_hook",
        "xdrl.trainer_hooks.logging.WandbFinishHook",
    ]
    assert cfg_gym_dqn.training_env.base_env.env_name == "CartPole-v1"
    assert cfg_mogym_ppo.training_scalarize_reward._target_ == "torchrl.envs.transforms.transforms.LineariseRewards"
    assert cfg_mogym_ppo.trainer.add_gae is False
