
class Train_data:
    def __init__(self,
                learning_rate,
                batch_size,
                learning_times,
                epsilon_ppo_clip,
                loss_entropy,
                gamma,
                n_step_return,
                eps_greedy_epsilon_start,
                eps_greedy_epsilon_end,
                learning_game_timing,
                save_epoch_timing):


        self.learning_rate = learning_rate          # learning rate
        self.batch_size = batch_size                # batch size
        self.learning_times = learning_times        # learn learning_times each 

        self.epsilon_ppo_clip = epsilon_ppo_clip    # ppo clip

        self.loss_entropy = loss_entropy            # entropy coefficient

        # params of advantage Bellman
        self.gamma = gamma
        self.n_step_return = n_step_return

        # eps_greedy
        self.eps_greedy_epsilon_start = eps_greedy_epsilon_start
        self.eps_greedy_epsilon_end = eps_greedy_epsilon_end

        self.learning_game_timing = learning_game_timing
        self.save_epoch_timing = save_epoch_timing