import time

def simple_run_loop(simple_env, simple_agent, max_frames=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    start_time = time.time()

    try:
        while True:
            model_features = simple_env.reset()
            while True:
                total_frames += 1
                actions = simple_agent.step(model_features)

                if max_frames and total_frames >= max_frames:
                    return
                if simple_env.last:
                    break
                feedback = simple_env.step(actions)
                model_features = feedback.features
                reward = feedback.reward
                if reward != 0:
                    print(reward)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
