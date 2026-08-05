Tutorials
=========

Use one ``RuntimeInteractionContext`` for each semantic model call. The
TensorDict stays owned by TorchRL.

Collection
----------

For local synchronous collection, pass the context directly as the policy::

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

Evaluation
----------

Use a separate evaluation contract with deterministic exploration and
``module_training=False``::

    with torch.no_grad():
        evaluation_rollout = env.rollout(1_000, policy=evaluation)

Losses
------

Call a replay-batch loss context in the same way::

    loss_td = replay_loss(replay_buffer.sample())

Keep the context open when hooks need to observe backward::

    optimiser.zero_grad()
    with replay_loss:
        loss_td = replay_loss.invoke(replay_buffer.sample())
        loss_td["loss_objective"].backward()
    optimiser.step()

Use a distinct ``TARGET`` contract for target/value estimation. Worker-copied,
asynchronous, and distributed collector policies are not supported.
