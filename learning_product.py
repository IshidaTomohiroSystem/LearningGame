from worker_thread import Worker_thread
from mode import ModeData
from game_product import Game_product
import tensorflow as tf
import threading
from mode import ModeData
import time
import numpy as np


class Learning_product(Game_product):
    def __init__(self, game_type, games, brain, mlagents, action_type, max_epochs, data):
        self.running_threads = []
        self.coord = tf.train.Coordinator()
        self.game_end = 0
        self.threads = []
        self.brain = brain
        self.games = games
        self.save_epoch_timing = data.save_epoch_timing

        self.data_path = 'data/input_data.npz'
        self.input_array = np.load(self.data_path)

        super(Learning_product, self).__init__(game_type, games, brain, mlagents, action_type, max_epochs, data)
        

    def create(self):
        super(Learning_product, self).create()
        with tf.device("/cpu:0"):
            
            for i in range (self.games):
                thread_name = "local_thread" + str(i + 1)
                thread = Worker_thread(thread_name, ModeData.learning, self.brain, i, self.mlagents, self.action_type, self.max_epochs, self.data)
                self.threads.append(thread)

        self.brain.sess.run(tf.global_variables_initializer())
        self.brain.save_initialize()


    def run(self):
        def work(worker):
            worker.run()
            self.game_end += 1

        for worker in self.threads:
            t = threading.Thread(target=work, args=(worker,), daemon=True)
            t.start()
            self.running_threads.append(t)
        #self.coord.join(self.running_threads)

        epoch = 0
        save_epochs = 0

        while True:
            if epoch != self.sum_epoch():

                if (epoch - save_epochs) >= self.save_epoch_timing:
                    self.brain.save_model()
                    save_epochs = epoch
                    print("epoch count : ", epoch)
                    self.predict_show()

            epoch = self.sum_epoch()
        
            if self.game_end == self.games:
                self.brain.save_model()
                break
            time.sleep(0.02)

    def sum_epoch(self):
        sum = 0
        for i in range(0, len(self.threads)):
            sum = sum + self.threads[i].epoch
        return sum


    def predict_show(self):
        print("===============================")
        print("")
        print("player predict")
        p1 = self.brain.predict_p(self.input_array['arr_0'])
        p1 = np.array([p1])
        print("action predict: ")
        print("Neutral : " + str(p1[0][0][0]))
        print("Right   : " + str(p1[0][0][1]))
        print("Left    : " + str(p1[0][0][2]))
        print("Attack  : " + str(p1[0][0][3]))
        print("Jump    : " + str(p1[0][0][4]))
        #print("Counter : " + str(p1[0][0][5]))
        #print("Strike  : " + str(p1[0][0][6]))
        #print("Shot    : " + str(p1[0][0][7]))
        print("")
        print("-------------------------------")
        print("")
        print("enemy predict")
        p2 = self.brain.predict_p(self.input_array['arr_1'])
        p2 = np.array([p2])
        print("action predict: ")
        print("Neutral : " + str(p2[0][0][0]))
        print("Right   : " + str(p2[0][0][1]))
        print("Left    : " + str(p2[0][0][2]))
        print("Attack  : " + str(p2[0][0][3]))
        print("Jump    : " + str(p2[0][0][4]))
        #print("Counter : " + str(p2[0][0][5]))
        #print("Strike  : " + str(p2[0][0][6]))
        #print("Shot    : " + str(p2[0][0][7]))
        print("")
        print("===============================")