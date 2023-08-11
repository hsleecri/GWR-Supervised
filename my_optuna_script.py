import gtls
from GammaGWR import GammaGWR
import optuna

def objective(trial):
    # Define the hyperparameter search space
    epochs = trial.suggest_int('epochs', 10, 50)
    a_threshold = trial.suggest_float('a_threshold', 0.1, 0.5)
    beta = trial.suggest_float('beta', 0.1, 1.0)
    epsilon_b = trial.suggest_float('epsilon_b', 0.01, 0.5)
    epsilon_n = trial.suggest_float('epsilon_n', 0.001, 0.1)

    # Create and train network with suggested hyperparameters
    my_net = GammaGWR()
    my_net.init_network(ds=ds_iris, random=False, num_context=1)
    my_net.train_ggwr(ds=ds_iris, epochs=epochs, a_threshold=a_threshold, beta=beta, l_rates=[epsilon_b, epsilon_n])

    # Test network
    my_net.test_gammagwr(ds_iris, test_accuracy=True)

    return 1 - my_net.test_accuracy  # Optuna minimizes the objective function

if __name__ == "__main__":
    # Import dataset from file
    data_flag = True

    if data_flag:
        ds_iris = gtls.Dataset(file='C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_레이블_첫행제거_레이블값1뺐음.csv', normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    study = optuna.create_study(direction="maximize")  # Maximize test accuracy
    study.optimize(objective, n_trials=50)  # Run 50 optimization trials

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", 1 - trial.value)  # Test accuracy
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
