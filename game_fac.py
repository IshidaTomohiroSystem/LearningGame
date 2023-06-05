from mode import ModeData
from learning_product import Learning_product
from test_product import Test_product
from load_learning_product import Load_learning_product
from battle_product import Battle_product

from ml_agents_load import MLAgentsLoad
from brain import Brain

class Game_factory:
    def __init__(self, games):
        self.games = games

    def create(self, game_type, action_type, max_epochs, data):
        self.mlagents = MLAgentsLoad(self.games, game_type)
        self.brain = Brain(self.mlagents, data, self.games)
        game = self.createGame(game_type, action_type, max_epochs, data)
        return game
    
    def createGame(self, game_type, action_type, max_epochs, data):
        if game_type == ModeData.learning:
            game = Learning_product(game_type, self.games, self.brain, self.mlagents, action_type, max_epochs, data)
            return game

        if game_type == ModeData.test:
            game = Test_product(game_type, self.games, self.brain, self.mlagents, action_type, max_epochs, data)
            return game

        if game_type == ModeData.load_learning:
            game = Load_learning_product(game_type, self.games, self.brain, self.mlagents, action_type, max_epochs, data)
            return game
    
        if game_type == ModeData.battle:
            game = Battle_product(game_type, self.games, self.brain, self.mlagents, action_type, max_epochs, data)
            return game