Tutorials
=========

TorchRL execution boundaries
-----------------------------

Build one ``RuntimeInteractionContext`` for each semantic model invocation.
The descriptor distinguishes collection, evaluation, loss, and target calls;
the live TensorDict remains owned by TorchRL. Assuming ``collection`` is a
context whose descriptor has phase ``COLLECTION``, direct execution is simply::

    action_td = collection(step_td)

The same callable can be passed as the policy of a local synchronous
collector. XDRL neither changes the returned keys nor controls batching::

    from torchrl.collectors import SyncDataCollector

    collector = SyncDataCollector(
        env,
        policy=collection,
        frames_per_batch=256,
        total_frames=10_000,
    )
    for rollout in collector:
        replay_buffer.extend(rollout)

Do not use this pattern with a collector that copies the policy into workers.
Hooks registered in the main process are not claimed to exist in those copies.

For deterministic evaluation, use a distinct descriptor with
``phase=InteractionPhase.EVALUATION``, ``exploration_mode="deterministic"``,
and ``module_training=False``. Both exploration state and the exact train/eval
flags of the module tree are restored after every call, including failures::

    with torch.no_grad():
        evaluation_rollout = env.rollout(1_000, policy=evaluation)

A replay-batch loss module uses the same invocation API. If activation hooks
only need the forward pass, the one-shot form is sufficient::

    loss_td = replay_loss(replay_buffer.sample())

When gradient hooks must observe backward, keep the interaction open until
backward completes. The descriptor should use ``phase=InteractionPhase.LOSS``
or ``OPTIMISATION`` and ``gradient_enabled=True``::

    optimiser.zero_grad()
    with replay_loss:
        loss_td = replay_loss.invoke(replay_buffer.sample())
        loss_td["loss_objective"].backward()
    optimiser.step()

Target/value estimation should use its own ``TARGET`` descriptor and identity,
even when it invokes a module with shared parameters. Logging, checkpointing,
replay sampling, and optimiser scheduling stay outside the interaction.
