import tensorflow as tf

class config():
  

    # env specific
    action_dim = 7
    observation_dim = [64, 64, 4]

    evaluate = True

    # output config
    output_path = "../results/" + env_name + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "Policy_gradient.png"
    record_path = output_path
    record_freq = 5
    summary_freq = 1

    
    # model and training config
    num_batches = 100 # number of batches trained on
    batch_size = 50000 # number of steps used to compute each policy update, default is 1000
    max_ep_len = 50000 # maximum episode length
    learning_rate = 3e-2
    gamma = 1 # the discount factor
    use_baseline = True 
    normalize_advantage=True

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
