
from worker_thread import Worker_thread
from mode import ModeData
from game_product import Game_product
import tensorflow as tf
import threading
from mode import ModeData
import time

class Load_learning_product(Game_product):
    def __init__(self, game_type, games, brain, mlagents, action_type, max_epochs, data):
        self.running_threads = []
        self.coord = tf.train.Coordinator()
        self.threads = []
        self.game_end = 0
        self.brain = brain
        self.games = games
        self.save_epoch_timing = data.save_epoch_timing

        super(Load_learning_product, self).__init__(game_type, games, brain, mlagents, action_type, max_epochs, data)
        

    def create(self):
        super(Load_learning_product, self).create()
        with tf.device("/cpu:0"):
            
        
            for i in range (self.games):
                thread_name = "local_thread" + str(i + 1)
                self.threads.append(Worker_thread(thread_name, ModeData.learning, self.brain, i, self.mlagents, self.action_type, self.max_epochs, self.data))

        self.brain.save_initialize()
        self.brain.saver.restore(self.brain.sess, './model/sword_model')

        


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
                    print("epoch", epoch)

            epoch = self.sum_epoch()
        
            if self.game_end == self.games:
                self.brain.save_model()
                break
            time.sleep(1)

    def sum_epoch(self):
        sum = 0
        for i in range(0, len(self.threads)):
            sum = sum + self.threads[i].epoch
        return sum


    def predict_show(self):
        print("predict")