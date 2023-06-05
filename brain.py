import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない
import math
import copy

class Brain:
    def __init__(self, mlagents_data, data, games):
        self.sess = tf.Session()
        self.learning_rate = data.learning_rate
        self.mlagents_data = mlagents_data
        self.epsilon = data.epsilon_ppo_clip      # ppo clip range
        self.loss_entropy = data.loss_entropy     # entropy coefficient
        self.batch_size = data.batch_size

        # params of advantage Bellman
        self.gamma = data.gamma
        self.n_step_return = data.n_step_return
        self.gamma_n = self.gamma ** self.n_step_return
        
        self.learning_times = data.learning_times

        self.saver = []

        self.games = games
        self.thread_epochs = [0] * games
        self.max_epochs = 0


        with tf.name_scope("brain"):
            self.train_queue = ([[[], [], [], [], []], [[], [], [], [], []]]) # s, a, r, s', s' terminal mask
            K.set_session(self.sess)
            self.model = self.build_model()
            self.old_model = clone_model(self.model)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.prob_old = tf.Variable(1, dtype=tf.float32)
            self.graph = self.build_graph()
            self.loss_mean = 0
            self.reward_mean = 0
            

    def build_model(self):
        input = Input(batch_shape=(None, self.mlagents_data.state_size * self.mlagents_data.stack_state_size))
        dense_1 = Dense(300,
                        activation='relu',
                        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                        bias_initializer=tf.constant_initializer(0))(input)

        batch_dense_1 = BatchNormalization()(dense_1)

        dense_2 = Dense(300,
                        activation='relu',
                        kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
                        bias_initializer = tf.constant_initializer(0))(batch_dense_1)

        batch_dense_2 = BatchNormalization()(dense_2)

        dense_3 = Dense(300,
                        activation='relu',
                        kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
                        bias_initializer = tf.constant_initializer(0))(batch_dense_2)

        out_actions = Dense(self.mlagents_data.actions,
                            activation=tf.nn.softmax,
                            kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
                            bias_initializer = tf.constant_initializer(0))(dense_3)
        
        dense_critic = Dense(300,
                             activation='relu',
                             kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
                             bias_initializer = tf.constant_initializer(0))(input)
        out_value = Dense(1,
                          kernel_initializer = tf.random_normal_initializer(stddev = 0.02),
                          bias_initializer = tf.constant_initializer(0))(dense_critic)

        #weight = get_weithts();

        model = Model(inputs=[input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def build_graph(self):
        self.s_t = tf.placeholder(tf.float32, shape=(None, self.mlagents_data.state_size * self.mlagents_data.stack_state_size))     # placeholderは変数が格納される予定地となる
        self.a_t = tf.placeholder(tf.float32, shape=(None, self.mlagents_data.actions))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))              # not immediate, but discount n step reward

        self.prob, v = self.model(self.s_t)
        self.prob_old, v_old = self.old_model(self.s_t)
  
        self.advantage = tf.subtract(self.r_t, v, name='advantage')
        
        #ratio = tf.exp(tf.log(prob + 1e-10) - tf.log(self.prob_old + 1e-10))
        self.ratio = tf.exp((self.prob + 1e-10) - (self.prob_old + 1e-10))
        self.p1 = self.ratio * self.advantage

        self.p2 = tf.multiply(tf.clip_by_value(self.ratio, 1 - self.epsilon, 1 + self.epsilon), self.advantage)

        self.actor_loss = tf.abs(-tf.reduce_mean(tf.minimum(self.p1, self.p2)))

        self.critic_loss = tf.reduce_mean(tf.square(self.r_t - v))
        self.entropy = self.loss_entropy * tf.reduce_mean(-(self.prob * tf.log(self.prob + 1e-10)))

        self.loss_total = self.critic_loss + self.actor_loss + self.entropy
        

        minimize = self.opt.minimize(self.loss_total)

        self.old_model = clone_model(self.model)
        #self.test_old_prob = self.prob_old
        #self.prob_old = self.prob

        return minimize
        
    def calculate_reward(self, reward):
        new_reward = copy.deepcopy(reward)
        add_reward_count = 0
        updated = False

        for i in range(0, len(reward)):
            if reward[i] == 0.0:
                continue
            add_reward_count = i
            for j in range(0, i):
                new_reward[j] = reward[i] * 0.1 / i
            new_reward[add_reward_count] = reward[add_reward_count]
        #print("reward", reward)
        #print("reward shape", reward.shape)
        #print("new reward", new_reward)
        #print("new reward shape", new_reward.shape)
        if add_reward_count == 0:
            updated = False
        else:
            updated = True

        return new_reward, updated

    def update_parameter_server(self, agent_num):  # localBrainの勾配でParameterServerの重みを学習,更新
        if(agent_num == 0):
            if len(self.train_queue[agent_num][0]) < self.batch_size:    # データがたまっていない場合は更新しない
                return

        if(agent_num == 1):
            if len(self.train_queue[agent_num][1]) < self.batch_size:    # データがたまっていない場合は更新しない
                return

        bat_length = len(self.train_queue[agent_num][0])
        s, a, r, s_, s_mask = self.train_queue[agent_num]
        self.train_queue = ([[[], [], [], [], []], [[], [], [], [], []]])
        
        loss_total = 0
        for i in range(math.floor((bat_length) / self.learning_times)):
            s_bat = np.vstack(s[self.learning_times * i : self.learning_times * i + self.batch_size])
            a_bat = np.vstack(a[self.learning_times * i : self.learning_times * i + self.batch_size])
            r_bat = np.vstack(r[self.learning_times * i : self.learning_times * i + self.batch_size])
            s__bat = np.vstack(s_[self.learning_times * i : self.learning_times * i + self.batch_size])
            s_mask_bat = np.vstack(s_mask[self.learning_times * i : self.learning_times * i + self.batch_size])

            r_bat_cal, updated = self.calculate_reward(r_bat)
            if updated == False:
                continue

            # Nステップあとの状態s_から、その先得られるであろう時間割引総報酬vを求めます
            _, v = self.model.predict(s__bat)

            # n-1ステップあとまでの時間割引総報酬rに、nから先に得られるであろう総報酬vに割引n乗したものを足す
            r_bat_n = r_bat_cal + self.gamma_n * v * s_mask_bat # set v to where s_ is terminal state
            feed_dict = {self.s_t: s_bat, self.a_t: a_bat, self.r_t: r_bat_n}     # 重みの更新に使用するデータ

            minimize = self.graph            

            loss_total, _ = self.sess.run([
                        self.loss_total,
                        minimize], 
                        feed_dict)   # ParameterServerの重みを更新

        print("============================================================")            
        #print(p)
        print("loss_total", loss_total)
            
        #os.system('cls')
            #print(loss_total, actor_loss, critic_loss, advantage)
        
        #summary = tf.Summary()
        #if self.learning_time != 0:
        #    summary.value.add(tag='reward', simple_value = self.reward_mean / self.learning_time)
        #    summary.value.add(tag='loss_total', simple_value = self.loss_mean / self.learning_time)
        #
        #    summary_writer.add_summary(summary, frames)
        #    self.loss_mean = 0
        #    self.reward_mean = 0
        #    self.learning_time = 0


            #self.prob_old = self.prob
        
    def predict_p(self, s):    # 状態sから各actionの確率pベクトルを返します
        s = np.array([s])
        s = s.reshape(1, self.mlagents_data.state_size * self.mlagents_data.stack_state_size)
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_, agent_num):
        self.train_queue[agent_num][0].append(s)
        self.train_queue[agent_num][1].append(a)
        self.train_queue[agent_num][2].append(r)

        if s_ is None:
            self.train_queue[agent_num][3].append(NONE_STATE)
            self.train_queue[agent_num][4].append(0.)
        else:
            self.train_queue[agent_num][3].append(s_)
            self.train_queue[agent_num][4].append(1.)

    def save_initialize(self):
        self.saver = tf.train.Saver()

    def sum_epochs(self, thread_num, epochs):
        self.thread_epochs[thread_num] = epochs
        sum = 0

        for i in range(0, self.games):
            sum = sum + self.thread_epochs[i]
        
        #if sum % 1 == 0:
        #    self.save_model()
    
    def save_model(self):
        self.saver.save(self.sess, './model/sword_model')
        tf.train.write_graph(tf.get_default_graph(), './model', 'sword_model.nn', as_text=False)
        