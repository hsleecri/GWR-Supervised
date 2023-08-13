import gtls
from GammaGWR import GammaGWR
import optuna
import optuna.visualization as vis
import plotly.io as pio

def objective(trial):
    # Define the hyperparameter search space
    epochs = trial.suggest_int('epochs', 1, 100)
    a_threshold = trial.suggest_float('a_threshold', 0.1, 0.5)
    beta = trial.suggest_float('beta', 0.01, 0.95)
    epsilon_b = trial.suggest_float('epsilon_b', 0.01, 0.5)
    epsilon_n = trial.suggest_float('epsilon_n', 0.001, 0.1)
    hab_threshold = trial.suggest_float('hab_threshold', 0.05, 0.2)
    tau_b = trial.suggest_float('tau_b', 0.1, 0.5)
    tau_n = trial.suggest_float('tau_n', 0.05, 0.2)
    max_age = trial.suggest_int('max_age', 100, 10000)
    num_context = trial.suggest_int('num_context', 1, 15)  # Adjust the range as needed

    # Create and train network with suggested hyperparameters
    my_net = GammaGWR()
    my_net.init_network(ds=ds_iris, random=False, num_context=num_context)
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
    study.optimize(objective, n_trials=500)  # Set optimization trials

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", 1 - trial.value)  # Test accuracy
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Visualization
    plot_optimization_history = vis.plot_optimization_history(study)
    plot_parallel_coordinate = vis.plot_parallel_coordinate(study)
    plot_slice = vis.plot_slice(study)

    # Save plots as PNG
    pio.write_image(plot_optimization_history, 'optimization_history.png')
    pio.write_image(plot_parallel_coordinate, 'parallel_coordinate.png')
    pio.write_image(plot_slice, 'slice_plot.png')