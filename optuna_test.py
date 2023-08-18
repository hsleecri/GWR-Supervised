import gtls
from GammaGWR_corrected import GammaGWR
import optuna
import optuna.visualization as vis
import plotly.io as pio
import os

def objective(trial, ds_iris,export_network_directory):
    """
    # Define the hyperparameter search space
    epochs = trial.suggest_int('epochs', 1, 100)
    a_threshold = trial.suggest_int('a_threshold', 10, 50) * 0.01
    beta = trial.suggest_int('beta', 1, 95) * 0.01
    epsilon_b = trial.suggest_int('epsilon_b', 10, 50) * 0.01
    epsilon_n = trial.suggest_int('epsilon_n', 1, 100) * 0.0001
    # hab_threshold = trial.suggest_int('hab_threshold', 5, 20) * 0.01
    # tau_b = trial.suggest_int('tau_b', 10, 50) * 0.01
    # tau_n = trial.suggest_int('tau_n', 5, 20) * 0.005
    max_age = trial.suggest_int('max_age', 100, 10000)
    num_context = trial.suggest_int('num_context', 1, 15)
    # penalty_weight = trial.suggest_int('penalty_weight', 1, 100)*0.001 #loss function에 penalty_weight를 넣고 싶으면 수정
    """
    # float로 하니까 값이 새버려서 step size만큼 이동이 안돼..
    #'''
    epochs = trial.suggest_int('epochs', 1, 100)
    a_threshold = trial.suggest_float('a_threshold', 0.4, 0.5)
    beta = trial.suggest_float('beta', 0.01, 0.95)
    epsilon_b = trial.suggest_float('epsilon_b', 0.1, 0.3)
    epsilon_n = trial.suggest_float('epsilon_n', 0.0005, 0.0015)
    hab_threshold = trial.suggest_float('hab_threshold', 0.05, 0.2)
    tau_b = trial.suggest_float('tau_b', 0.1, 0.5)
    tau_n = trial.suggest_float('tau_n', 0.05, 0.2)
    max_age = trial.suggest_int('max_age', 100, 10000)
    num_context = trial.suggest_int('num_context', 1, 15) #이거가 오류가 잘 나는데, features 개수보다 num_context보다 크면 오류가 남
    #'''

    # Create and train network with suggested hyperparameters
    my_net = GammaGWR()
    my_net.init_network(ds=ds_iris, random=False, num_context=num_context)
    my_net.train_ggwr(ds=ds_iris, epochs=epochs, a_threshold=a_threshold, beta=beta, l_rates=[epsilon_b, epsilon_n], hab_threshold=hab_threshold, tau_b=tau_b, tau_n=tau_n, max_age=max_age) #모든 하이퍼파라미터를 교정하고 싶을 때
    # my_net.train_ggwr(ds=ds_iris, epochs=epochs, a_threshold=a_threshold, beta=beta, l_rates=[epsilon_b, epsilon_n], hab_threshold=1, tau_b=0.3, tau_n=0.1, max_age=max_age) #성능에 직접적으로 큰 영향을 미치는 하이퍼파라미터를 교정할 때
    """
    기존 연구 코드에서 explicit하게 hyperparameter라고 써놓은 것들: num_context=1,epochs=15,a_threshold=0.85, beta=0.7, l_rates=[0.2, 0.001]
    기존 연구 코드에서 코드 내부에 특정값으로 설정되어 있던 것들: hab_threshold = 0.1, tau_b = 0.3, tau_n = 0.1, max_neighbors = 6, max_age = 600
    내가 생각할 때 성능에 큰 영향을 미치는 hyperparamter들: num_context, epochs, a_thrshold, beta, l_rates, max_age
    """
    # Export Network
    file_name = f'trial_{trial.number}.pkl'
    gtls.export_network_to_local(file_name, my_net, export_network_directory)

    # Test network
    my_net.test_gammagwr(ds_iris, test_accuracy=True)

    '''
    # Get the number of nodes
    num_nodes = my_net.get_num_nodes()

    # Compute the loss, with a penalty if num_nodes is not close to the number of classes
    loss = 1 - my_net.test_accuracy + abs(num_nodes - my_net.num_classes) * 0.01
    '''
    loss = 1- my_net.test_accuracy #loss funcntion을 test_accuracy로 설정
    # Attach the test accuracy to the trial
    trial.set_user_attr('test_accuracy', my_net.test_accuracy)

    print(f"Trial {trial.number} finished with test accuracy: {my_net.test_accuracy:.6f}")

    return loss

def main(data_file, output_directory, export_network_directory, n_trials):
    # Import dataset from file
    ds_iris = gtls.Dataset(file=data_file, normalize=True)
    print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    study = optuna.create_study(direction="minimize")  # Minimizing loss = Maximize test accuracy
    study.optimize(lambda trial: objective(trial, ds_iris,export_network_directory), n_trials=n_trials)  # Set optimization trials

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)  # Loss
    print("Test Accuracy: ", trial.user_attrs['test_accuracy'])  # Print the test_accuracy
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Visualization
    plot_optimization_history = vis.plot_optimization_history(study)
    plot_parallel_coordinate = vis.plot_parallel_coordinate(study)
    plot_slice = vis.plot_slice(study)

    gtls.import_network()
    # Save plots as PNG to the specified directory
    pio.write_image(plot_optimization_history, output_directory + 'optimization_history.png')
    pio.write_image(plot_parallel_coordinate, output_directory + 'parallel_coordinate.png')
    pio.write_image(plot_slice, output_directory + 'slice_plot.png')

if __name__ == "__main__":
    data_file = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_표준편차정렬_레이블_27개feature_첫행제거_레이블값1뺐음.csv'
    output_directory = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\PNG\\'
    export_network_directory = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\Pickle\\' 
    main(data_file, output_directory, export_network_directory, n_trials=1) #set the number of trials in study