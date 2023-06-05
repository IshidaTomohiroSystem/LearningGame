from worker_thread import Worker_thread
from game_product import Game_product
from mode import ModeData
import tensorflow as tf


class Test_product(Game_product):
    def __init__(self, game_type, games, brain, mlagents, action_type, max_epochs, data):
        self.running_thread = []
        super(Test_product, self).__init__(game_type, games, brain, mlagents, action_type, max_epochs, data)

    def create(self):
        self.brain.sess.run(tf.global_variables_initializer())
        self.brain.save_initialize()
        self.brain.saver.restore(self.brain.sess,'./model/sword_model')

        self.running_thread = Worker_thread("test_thread", ModeData.test, self.brain, 0, self.mlagents, self.action_type, self.max_epochs, self.data)

    def run(self):
        self.running_thread.run()
