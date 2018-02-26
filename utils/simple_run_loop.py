import time

def simple_run_loop(simple_env, simple_agent, max_frames=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    start_time = time.time()

    try:
        while True:

            while True:
                total_frames += 1
                model_features = simple_env.get_features()

                actions = simple_agent.step(model_features)

                if max_frames and total_frames >= max_frames:
                    return
                if simple_env.last:
                    break
                model_features = simple_env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))