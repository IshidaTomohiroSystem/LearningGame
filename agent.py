import random
import numpy as np

class Agent:
    def __init__(self, brain, mlagents_data, data):
        self.brain = brain
        self.memory = [[],[]]   # s, a, r, s_の保存メモリ  used for n_step return
        self.r_sum = 0.                      # 時間割引した今からNステップ分後までの総報酬R
        self.select_action = [0] * mlagents_data.actions
        self.actions = mlagents_data.actions

        self.eps_start =data.eps_greedy_epsilon_start
        self.eps_end = data.eps_greedy_epsilon_end

        # params of advantage Bellman
        self.gamma = brain.gamma
        self.n_step_return = brain.n_step_return
        self.gamma_n = brain.gamma_n

        self.sum_chosen_count = 0
        self.chosen_count = np.zeros(self.actions)
        self.estimated_rewards = np.zeros(self.actions)

    def act_random(self, s, epochs, max_epoch):
        return random.randint(0, self.actions -1)    # ランダムに行動

    def calculate_u(self, act_num):
        if self.chosen_count[act_num] == 0:
            return 10.0
        else:
            return np.sqrt(self.sum_chosen_count) / (self.chosen_count[act_num] + 1)

    def act_ucb_learning(self, s, epochs, max_epoch):
        upper_bound_probs = []
        for i in range(self.actions):
            u = self.calculate_u(i)
            prob = u + self.estimated_rewards[i]
            upper_bound_probs.append(prob)
        action = np.argmax(upper_bound_probs)

        s = np.array([s])
        p = self.brain.predict_p(s)
        p = np.array([p])
        p = p.reshape(self.actions)

        reward = np.random.binomial(n=1, p=p[action])
        self.chosen_count[action] += 1
        self.sum_chosen_count += 1
        return action

    def act_eps_greedy_learing(self, s, epochs, max_epoch):

        eps = self.eps_start + epochs * (self.eps_end - self.eps_start) / max_epoch
        if random.random() < eps:
            return random.randint(0, self.actions -1)    # ランダムに行動

        else:
            s = np.array([s])
            p = self.brain.predict_p(s)
            p = np.array([p])
            p = p.reshape(self.actions)
        
            a = np.random.choice(self.actions, p=p)
            # probability = pのコードだと確率p[0]にしたがって行動を選択
            # pにはいろいろな情報が入っているが確率のベクトルは要素の0番目

            return a


    def act_eps_greedy_test(self, s, epochs, max_epoch):
        s = np.array([s])
        p = self.brain.predict_p(s)
        p = np.array([p])
        p = p.reshape(self.actions)

        a = np.argmax(p)
        
        return a

    def advantage_push_brain(self, s, a, r, s_, agent_num):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n -1]
            return s, a, self.r_sum, s_

        # one hot a_cat -> add memory
        a_cats = np.zeros(self.actions)
        a_cats[a] = 1
        self.memory[agent_num].append((s, a_cats, r, s))

        # use previous step
        self.r_sum = (self.r_sum + r * self.gamma_n) / self.gamma

        # advantage input
        if s_ is None:
            while len(self.memory[agent_num]) > 0:
                n = len(self.memory[agent_num])
                s, a, r, s_ = get_sample(self.memory[agent_num], n)
                self.brain.train_push(s, a, r, s_, agent_num)
                self.r_sum = (self.r_sum - self.memory[agent_num][0][2]) / self.gamma
                slef.memory[agent_num].pop(0)
            self.r_sum = 0  # next r = 0

        if len(self.memory[agent_num]) >= self.n_step_return:
            s, a, r, s_ = get_sample(self.memory[agent_num], self.n_step_return)
            self.brain.train_push(s, a, r, s_, agent_num)
            self.r_sum = self.r_sum - self.memory[agent_num][0][2]
            self.memory[agent_num].pop(0)

        