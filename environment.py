import numpy as np
from agent import Agent
from mode import ModeData, ActionSelectType

class Environment:
    def __init__(self, name, thread_type, brain, game_num, mlagents_data, action_type, max_epochs, data):
        self.name = name
        self.thread_type = thread_type
        self.brain = brain
        self.env = mlagents_data.env_mlagents[game_num]
        self.agent = []
        self.game_num = game_num

        self.action_data = []
        
        self.env_info_player = mlagents_data.player_env[self.game_num]
        self.env_info_enemy = mlagents_data.enemy_env[self.game_num]

        self.step = 0
        self.player_brain_name = mlagents_data.env_mlagents[0].brain_names[0]
        self.enemy_brain_name = mlagents_data.env_mlagents[0].brain_names[1]
        
        self. agent_size = mlagents_data.agent_size

        for i in range(0, self.agent_size):
            self.agent.append(Agent(brain, mlagents_data, data))

        self.env_enfo = []
        self.total_reward_vec = np.zeros(10)
        self.count_trial_each_thread = 0

        self.action_update_times = 4

        if thread_type == ModeData.learning or thread_type == ModeData.load_learning:
            self.run_method = self.run_learning

        if thread_type == ModeData.test:
            self.run_method = self.run_test

        if thread_type == ModeData.battle:
            self.run_method = self.run_battle

        self.action_type = action_type

        if self.action_type == ActionSelectType.random:
            self.player_action_method = self.agent[0].act_random
            self.enemy_action_method = self.agent[1].act_random

        if self.action_type == ActionSelectType.eps_greedy:
            self.player_action_method = self.agent[0].act_eps_greedy_learing
            self.enemy_action_method = self.agent[1].act_eps_greedy_learing

        if self.action_type == ActionSelectType.eps_greedy_test:
            self.player_action_method = self.agent[0].act_eps_greedy_test
            self.enemy_action_method = self.agent[1].act_eps_greedy_test

        if self.action_type == ActionSelectType.ucb:
            self.player_action_method = self.agent[0].act_ucb_learning
            self.enemy_action_method = self.agent[1].act_ucb_learning

        self.games = 1
        self.epochs = 0
        self.learning_game_timing = data.learning_game_timing
        self.max_epochs = max_epochs


    def run(self):
        epochs = self.run_method()
        return epochs

    def run_learning(self):
        player_action = 0
        enemy_action = 0

        state = []
        reward = []

        state.append(self.env_info_player.vector_observations[0])
        state.append(self.env_info_enemy.vector_observations[0])

        state = np.array(state)

        for i in range(0, self. agent_size):
            reward.append(0)

        while True:
            action = []
            player_state = self.env_info_player.vector_observations[0]
            enemy_state = self.env_info_enemy.vector_observations[0]

            #if self.step % self.action_update_times == 0:
            player_action = self.player_action_method(player_state, self.epochs, self.max_epochs)
            #enemy_action = self.enemy_action_method(enemy_state, self.epochs, self.max_epochs)
            enemy_action = 0

            action.append(player_action)   # 行動を決定
            action.append(enemy_action)   # 行動を決定
                
            select_action = {self.player_brain_name : action[0], self.enemy_brain_name : action[1]}
            env_info_ = self.env.step(select_action)  # 行動を実施

            state_ = []
            reward_ = []
            done = []

            # input data state reward game_end
            state_.append(env_info_[self.player_brain_name].vector_observations)
            reward_.append(env_info_[self.player_brain_name].rewards[0])
            done.append(env_info_[self.player_brain_name].local_done[0])

           
            #state_.append(env_info_[self.enemy_brain_name].vector_observations)
            #reward_.append(env_info_[self.enemy_brain_name].rewards[0])
            #done.append(env_info_[self.enemy_brain_name].local_done[0])

            # with advantage state reward
            #self.agent[0].advantage_push_brain(state[0], action[0], reward_[0], state_[0], 0)
            #self.agent[0].advantage_push_brain(state[1], action[1], reward_[1], state_[1], 1)

            self.env_enfo = env_info_

            for i in range(0, 1):
                state[i] = state_[i]
                reward[i] += reward_[i]

            #for i in range(0, self. agent_size):
            #    state[i] = state_[i]
            #    reward[i] += reward_[i]

            # with advantage state reward
            self.agent[0].advantage_push_brain(state[0], action[0], reward_[0], state_[0], 0)
            #self.agent[0].advantage_push_brain(state[1], action[1], reward_[1], state_[1], 1)

            if any(done) and self.games % self.learning_game_timing == 0:
                self.agent[0].brain.update_parameter_server(0)
                #self.agent[1].brain.update_parameter_server(1)
                self.epochs += 1
                print("environment epochs : ", self.epochs)
                print("environment games : ", self.games)
                print("learning time : ", self.learning_game_timing)

            if any(done):
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], reward))
                self.count_trial_each_thread += 1
                self.env.reset(train_mode=True)[self.player_brain_name]
                self.env.reset(train_mode=True)[self.enemy_brain_name]
                self.games += 1
                break
            self.step += 1

        return self.epochs

    def run_test(self):
        player_action = 0
        enemy_action = 0

        state = []
        reward = []
        done = []

        state.append(self.env_info_player.vector_observations[0])
        state.append(self.env_info_enemy.vector_observations[0])

        state = np.array(state)

        for i in range(0, self. agent_size):
            reward.append(0)

        self.env.reset(train_mode=False)[self.player_brain_name]
        self.env.reset(train_mode=False)[self.enemy_brain_name]

        while True:
            action = []
            player_state = self.env_info_player.vector_observations[0]
            enemy_state = self.env_info_enemy.vector_observations[0]

            if self.step % self.action_update_times == 0:
                player_action = self.player_action_method(player_state, self.epochs, self.max_epochs)
                enemy_action = 0
                #enemy_action = self.enemy_action_method(enemy_state, self.epochs, self.max_epochs)

            action.append(player_action)   # 行動を決定
            action.append(enemy_action)   # 行動を決定
                
            select_action = {self.player_brain_name : action[0], self.enemy_brain_name : action[1]}
            env_info_ = self.env.step(select_action)  # 行動を実施

            done.append(env_info_[self.player_brain_name].local_done[0])
            done.append(env_info_[self.enemy_brain_name].local_done[0])

            if any(done):
                self.epochs += 1
                break
        return self.epochs

    def run_battle(self):
        player_action = 0
        enemy_action = 0

        state = []
        reward = []
        done = []

        state.append(self.env_info_player.vector_observations[0])
        state.append(self.env_info_enemy.vector_observations[0])

        state = np.array(state)

        for i in range(0, self. agent_size):
            reward.append(0)

        self.env.reset(train_mode=False)[self.player_brain_name]
        self.env.reset(train_mode=False)[self.enemy_brain_name]

        while True:
            action = []
            enemy_state = self.env_info_enemy.vector_observations[0]

            if self.step % self.action_update_times == 0:
                enemy_action = self.enemy_action_method(enemy_state, self.epochs, self.max_epochs)

            action.append(enemy_action)   # 行動を決定
                
            select_action = {self.enemy_brain_name : action[0]}
            env_info_ = self.env.step(select_action)  # 行動を実施

            done.append(env_info_[self.player_brain_name].local_done[0])
            done.append(env_info_[self.enemy_brain_name].local_done[0])

            if any(done):
                self.epochs += 1
                break
        return self.epochs