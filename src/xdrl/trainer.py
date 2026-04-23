from __future__ import annotations

import pathlib
from collections.abc import Callable
from functools import partial

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import optim

from torchrl.collectors import BaseCollector
from torchrl.data import ReplayBuffer
from torchrl.modules import EGreedyModule
from torchrl.objectives import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers.common import Logger
from torchrl.trainers.trainers import ReplayBufferTrainer, TargetNetUpdaterHook, Trainer, UpdateWeights


def _process_batch_for_qmix(batch: TensorDictBase, *, group: str) -> TensorDictBase:
    keys = set(batch.keys(True, True))

    if ("next", "done") not in keys and ("next", group, "done") in keys:
        batch.set(("next", "done"), batch.get(("next", group, "done")).any(dim=-2))

    if ("next", "terminated") not in keys and ("next", group, "terminated") in keys:
        batch.set(("next", "terminated"), batch.get(("next", group, "terminated")).any(dim=-2))

    if ("next", "reward") not in keys and ("next", group, "reward") in keys:
        batch.set(("next", "reward"), batch.get(("next", group, "reward")).mean(dim=-2))

    return batch


class QmixTrainer(Trainer):
    """Trainer for off-policy QMIX with replay-buffer updates and target syncing."""

    def __init__(
        self,
        *,
        collector: BaseCollector,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        replay_buffer: ReplayBuffer | None = None,
        target_net_updater: TargetNetUpdater | None = None,
        greedy_module: EGreedyModule | None = None,
        group: str = "agents",
        weights_source=None,
        async_collection: bool = False,
        log_timings: bool = False,
    ) -> None:
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            async_collection=async_collection,
            log_timings=log_timings,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection
        self.group = group

        self.register_op("batch_process", partial(_process_batch_for_qmix, group=group))

        if replay_buffer is not None:
            rb_trainer = ReplayBufferTrainer(
                replay_buffer,
                batch_size=None,
                flatten_tensordicts=True,
                memmap=False,
                device=getattr(replay_buffer.storage, "device", "cpu"),
                iterate=True,
            )
            if not self.async_collection:
                self.register_op("pre_epoch", rb_trainer.extend)
            self.register_op("process_optim_batch", rb_trainer.sample)
            self.register_op("post_loss", rb_trainer.update_priority)

        if target_net_updater is not None:
            self.register_op("post_optim", TargetNetUpdaterHook(target_net_updater))

        self.greedy_module = greedy_module
        if greedy_module is not None:
            self._greedy_last_frames = 0
            self.register_op("post_steps", self._step_greedy)

        if weights_source is None:
            weights_source = getattr(self.loss_module, "local_value_network", None)
            if weights_source is None:
                msg = "QmixTrainer could not infer weights_source from loss_module.local_value_network."
                raise RuntimeError(msg)
            if greedy_module is not None:
                weights_source = TensorDictSequential(weights_source, greedy_module)

        policy_weights_getter = partial(TensorDict.from_module, weights_source)
        update_weights = UpdateWeights(self.collector, 1, policy_weights_getter=policy_weights_getter)
        self.register_op("post_steps", update_weights)

    def _step_greedy(self) -> None:
        if self.greedy_module is None:
            return
        delta = self.collected_frames - self._greedy_last_frames
        if delta > 0:
            self.greedy_module.step(delta)
            self._greedy_last_frames = self.collected_frames
