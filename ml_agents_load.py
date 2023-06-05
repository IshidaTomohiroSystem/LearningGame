from mlagents.envs import UnityEnvironment
from mode import ModeData
import numpy as np

class MLAgentsLoad:
    def __init__(self, game_num, mode):
        self.mode_data = mode
        #self.game_file_path = "C:/Users/ishida.tomohiro/Documents/project/unity_project/sword_ver2/BuildLearning/sword_ver2.exe"    # game file path
        #self.game_battle_file_path = "C:/Users/ishida.tomohiro/Documents/project/unity_project/sword_ver2/BuildBattle/sword_ver2.exe"
        
        self.game_file_path = "C:/Users/ishida.tomohiro/Documents/project/unity_project/sword_ver2/BuildActionSelect/sword_ver2.exe"    # game file path
        self.game_test_file_path = "C:/Users/ishida.tomohiro/Documents/project/unity_project/sword_ver2/BuildActionSelectTest/sword_ver2.exe"
        self.game_battle_file_path = "C:/Users/ishida.tomohiro/Documents/project/unity_project/sword_ver2/BuildActionSelectBattle/sword_ver2.exe"

        self.games = game_num           # thread num
        self.env_mlagents = []          # game env
        self.random_seed = 1            # random seed
        self.brain_name = []            # brain name
        self.mlagents_player_brain = [] # player brain
        self.mlagents_enemy_brain = []  # enemy brain
        self.player_env = []            # player env(input, reward, etc)
        self.enemy_env = []             # enmy env(input, reward, etc)
        self.agent_size = 2             # agent size

        self.actions = 0                # action size
        self.state_size = 0             # state size 
        self.none_state = []            # state initial
        self.stack_state_size = 0       # stack state size

        self.env_load()
        self.name_load()
        self.brain_load()
        self.env_chara_load()

    def env_load(self):
        file_path = ""
        if self.mode_data == ModeData.test:
            file_path = self.game_test_file_path
        elif self.mode_data == ModeData.battle:
            file_path = self.game_battle_file_path
        else:
            file_path = self.game_file_path
        for i in range(self.games):
            self.env_mlagents.append(UnityEnvironment(file_name = file_path, worker_id=i, seed=self.random_seed))

    def name_load(self):
        self.brain_name.append(self.env_mlagents[0].brain_names[0])     # player brain name
        self.brain_name.append(self.env_mlagents[0].brain_names[1])     # enemy brain name

    def brain_load(self):
        for i in range(self.games):
            self.mlagents_player_brain.append(self.env_mlagents[i].brains[self.brain_name[0]])   # player brain
            self.mlagents_enemy_brain.append(self.env_mlagents[i].brains[self.brain_name[1]])    # enemy brain
    
    def env_chara_load(self):
        for i in range(self.games):
            self.player_env.append(self.env_mlagents[i].reset(train_mode=True)[self.brain_name[0]]) # player env
            self.enemy_env.append(self.env_mlagents[i].reset(train_mode=True)[self.brain_name[1]])  # enemy env
        
        self.state_size = self.mlagents_player_brain[0].vector_observation_space_size;
        self.actions = self.mlagents_player_brain[0].vector_action_space_size[0]
        self.none_state = np.zeros(self.state_size)
        self.stack_state_size = self.mlagents_player_brain[0].num_stacked_vector_observations

    def close(self):
        for i in range(self.games):
            self.env_mlagents[i].close()