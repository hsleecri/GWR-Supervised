import gtls
from GammaGWR import GammaGWR
import optuna
import optuna.visualization as vis
import plotly.io as pio

def objective(trial, ds_iris):
    '''
    epochs = trial.suggest_int('epochs', 1, 100)
    a_threshold = trial.suggest_int('a_threshold', 1, 5) * 0.1
    beta = trial.suggest_int('beta', 1, 95) * 0.01
    epsilon_b = trial.suggest_int('epsilon_b', 1, 50) * 0.01
    epsilon_n = trial.suggest_int('epsilon_n', 1, 100) * 0.001
    hab_threshold = trial.suggest_int('hab_threshold', 5, 20) * 0.01
    tau_b = trial.suggest_int('tau_b', 10, 50) * 0.01
    tau_n = trial.suggest_int('tau_n', 5, 20) * 0.005
    max_age = trial.suggest_int('max_age', 100, 10000, step=100)
    num_context = trial.suggest_int('num_context', 1, 15)
    '''
    # float로 하니까 값이 새버려서 step size만큼 이동이 안돼..
    # '''
    epochs = trial.suggest_int('epochs', 1, 100)
    a_threshold = trial.suggest_float('a_threshold', 0.1, 0.5)
    beta = trial.suggest_float('beta', 0.01, 0.95)
    epsilon_b = trial.suggest_float('epsilon_b', 0.01, 0.5)
    epsilon_n = trial.suggest_float('epsilon_n', 0.001, 0.1)
    hab_threshold = trial.suggest_float('hab_threshold', 0.05, 0.2)
    tau_b = trial.suggest_float('tau_b', 0.1, 0.5)
    tau_n = trial.suggest_float('tau_n', 0.05, 0.2)
    max_age = trial.suggest_int('max_age', 100, 10000)
    num_context = trial.suggest_int('num_context', 1, 15)
    # '''

    my_net = GammaGWR()
    my_net.init_network(ds=ds_iris, random=False, num_context=num_context)
    my_net.train_ggwr(ds=ds_iris, epochs=epochs, a_threshold=a_threshold, beta=beta, l_rates=[epsilon_b, epsilon_n], hab_threshold=hab_threshold, tau_b=tau_b, tau_n=tau_n, max_age=max_age)

    my_net.test_gammagwr(ds_iris, test_accuracy=True)
    return my_net.test_accuracy

def main(num_trials=50):
    
    #File paths
    file_path = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_레이블_첫행제거_레이블값1뺐음.csv'
    result_file = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_레이블_첫행제거_레이블값1뺐음_결과.csv'
    output_directory = 'C:\\Users\\hslee\Desktop\dataset\\HYEONSU\\4공정\\PNG\\'

    # Flags
    data_flag = True
    import_flag = False
    train_flag = True
    test_flag = True
    export_flag = False
    result_flag = True
    plot_flag = True
    visualize_optimization_flag = True
    
    if data_flag:
        ds_iris = gtls.Dataset(file=file_path, normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    if import_flag:
        fname = 'my_net.ggwr'
        my_net = gtls.import_network(fname, GammaGWR)

    if train_flag:
        study = optuna.create_study(direction="maximize")  # Maximize test accuracy
        study.optimize(lambda trial: objective(trial, ds_iris), num_trials)  # Set optimization trials

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("Value: ", trial.value)  # Loss
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train model with best hyperparameters
        my_net = GammaGWR()
        # Initialization and training code with optimal hyperparameters
        my_net.init_network(ds=ds_iris, random=False, num_context=trial.params['num_context'])
        my_net.train_ggwr(ds=ds_iris, epochs=trial.params['epochs'], a_threshold=trial.params['a_threshold'] * 0.1,
                          beta=trial.params['beta'] * 0.01, l_rates=[trial.params['epsilon_b'] * 0.01, trial.params['epsilon_n'] * 0.001],
                          hab_threshold=trial.params['hab_threshold'] * 0.01, tau_b=trial.params['tau_b'] * 0.01, tau_n=trial.params['tau_n'] * 0.005,
                          max_age=trial.params['max_age'])

    if test_flag:
        my_net.test_gammagwr(ds_iris, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)

    if result_flag:
        gtls.export_result(result_file, my_net, ds_iris)

    if export_flag:
        fname = 'my_net.ggwr'
        gtls.export_network(fname, my_net)

    if plot_flag:
        gtls.plot_network_hs(my_net, edges=True, labels=True)
    
    if visualize_optimization_flag:
        # Visualization
        plot_optimization_history = vis.plot_optimization_history(study)
        plot_parallel_coordinate = vis.plot_parallel_coordinate(study)
        plot_slice = vis.plot_slice(study)

        # Directory to save plots
        output_directory = r'C:\\Users\\hslee\Desktop\dataset\\HYEONSU\\4공정\\PNG\\'

        # Save plots as PNG to the specified directory
        pio.write_image(plot_optimization_history, output_directory + 'optimization_history.png')
        pio.write_image(plot_parallel_coordinate, output_directory + 'parallel_coordinate.png')
        pio.write_image(plot_slice, output_directory + 'slice_plot.png')


if __name__ == "__main__":
    main()
