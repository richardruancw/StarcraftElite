import tensorflow as tf

class config():
  

    # env specific

    evaluate = False

    # output config
    output_path = "../results/policy_gradient/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "Policy_gradient.png"
    summary_freq = 5
    eval_freq = 5
    save_freq = 5

    
    # model and training config
    mode = "TD" # value can be "TD" or "MC"
    num_batches = 100 # number of batches trained on
    batch_size = 20 # number of steps used to compute each policy update, default is 1000
    max_ep_len = 20 # maximum episode length
    eval_batch_size = batch_size
    rand_begin = 0.2
    rand_end = 0
    rand_steps = num_batches
    learning_rate = 3e-2
    gamma = 0.9 # the discount factor
    use_baseline = True 
    normalize_advantage = True
    history_mul = 4
    restore = False

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    assert rand_steps <= num_batches
