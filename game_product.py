from worker_thread import Worker_thread


class Game_product:
    def __init__(self, game_type, games, brain, mlagents, action_type, max_epochs, data):
        self.game_type = game_type
        self.games = games
        self.brain = brain
        self.mlagents = mlagents
        self.action_type = action_type
        self.max_epochs = max_epochs
        self.data = data
        self.create()

    def create(self):
        self.createGame()
    
    def createGame(self):
        print("create game")
        pass

    def run(self):
        pass


