import enum
class ModeData(enum.Enum):
    learning = "Learning"
    load_learning = "Load_learning"
    test = "Test"
    battle = "Battle"

class ActionSelectType(enum.Enum):
    random = "Random"
    eps_greedy = "Epsilon_Greedy"
    eps_greedy_test = "Test"
    ucb = "UCB"
