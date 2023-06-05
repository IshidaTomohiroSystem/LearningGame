from mode import ModeData, ActionSelectType
from game_fac import Game_factory
from train_data import Train_data

def main():

    games = 1
    mode = ModeData.test
    action = ActionSelectType.eps_greedy_test
    max_epochs = 10000



    data = Train_data(
        learning_rate               = 0.00002,
        batch_size                  = 32,
        learning_times              = 32,
        epsilon_ppo_clip            = 0.02,
        loss_entropy                = 0.001,
        gamma                       = 1,
        n_step_return               = 1,
        eps_greedy_epsilon_start    = 0.2,
        eps_greedy_epsilon_end      = 0.0,
        learning_game_timing        = 1,       # epoch = game count / learning game timimg
        save_epoch_timing           = 1)

    fac = Game_factory(games)
    test = fac.create(mode, action, max_epochs, data)

    test.run()

    
    #fac.mlagents.close()
    print("eoriwori")

if __name__ == "__main__":
    main();