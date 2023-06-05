from environment import Environment
from mode import ModeData

class Worker_thread:
    def __init__(self, thread_name, thread_type, brain, game_num, mlagents_data, action_type, max_epochs, data):
        self.environment = Environment(thread_name, thread_type, brain, game_num, mlagents_data, action_type, max_epochs, data)
        self. thread_type = thread_type
        self.max_epochs = max_epochs
        self.brain = brain
        self.game_num = game_num
        self.epoch = 0

    def run(self):
        while True:
            self.epoch = self.environment.run()
            self.brain.sum_epochs(self.game_num, self.epoch)

            
            if self.max_epochs <= self.epoch:
                break

    
