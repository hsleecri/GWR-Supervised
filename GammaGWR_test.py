import gtls
from GammaGWR import GammaGWR

if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network
    import_flag = True
    # Train GWR with imported dataset    
    train_flag = False
    # Compute classification accuracy    
    test_flag = True
    # Export pickled network     
    export_flag = False
    # Show result data
    result_flag = True
    # Compare the result with orignial data
    compare_flag = True
    # Show result data like Temporal Action Segmentation
    result_segmentation_flag = True
    # Compare the result with orignial data like Temporal Action Segmentation
    compare_segmentation_flag = True    
    # Plot network (2D projection)
    plot_flag = True

    
    if data_flag:
        ds_iris = gtls.Dataset(file='C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_표준편차정렬_레이블_27개feature_첫행제거_레이블값1뺐음.csv', normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    if import_flag:
        num=116
        fname = ('C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\Pickle\\trial_{}.pkl'.format(num))
        my_net = gtls.import_network(fname, GammaGWR)
    
    if train_flag:
       # Create network 
       my_net = GammaGWR()
       # Initialize network with two neurons
       my_net.init_network(ds=ds_iris, random=False, num_context=13)
       # Train network on dataset
    #    my_net.train_ggwr(ds=ds_iris, epochs=15, a_threshold=0.35, beta=0.7, l_rates=[0.2, 0.001])
       my_net.train_ggwr(ds=ds_iris, epochs=86, a_threshold=0.1826903787085066, beta=0.06725872568741006,l_rates=[0.40323047009253504,0.0326872706770626], 
                         hab_threshold=0.09979474963903648, tau_b=0.3535307894839173, tau_n=0.13849675735445163, max_age=1030) # l_rates=[epsilon_b, epsilon_n]

    """
    기존 연구 코드에서 explicit하게 hyperparameter라고 써놓은 것들: num_context=1,epochs=15,a_threshold=0.85, beta=0.7, l_rates=[0.2, 0.001]
    기존 연구 코드에서 코드 내부에 특정값으로 설정되어 있던 것들: hab_threshold = 0.1, tau_b = 0.3, tau_n = 0.1, max_neighbors = 6, max_age = 600
    내가 생각할 때 성능에 큰 영향을 미치는 hyperparamter들: num_context, epochs, a_thrshold, beta, l_rates, max_age
    """

    if test_flag:
        my_net.test_gammagwr(ds_iris, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)
 
    if result_flag:
        fname= 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_표준편차정렬_레이블_27개feature_첫행제거_레이블값1뺐음_결과.csv'
        gtls.show_result(fname, my_net, ds_iris)

    if export_flag:
        fname = 'my_net.ggwr'
        gtls.export_network(fname, my_net)

    if compare_flag:
        gtls.show_original_clusters(ds_iris)

    if result_segmentation_flag:
        gtls.show_result_segmentation(my_net,ds_iris)

    if compare_segmentation_flag:
        gtls.show_original_clusters_segmentation(ds_iris)

    if plot_flag:
        gtls.plot_network_with_pca(my_net, edges=True, labels=True) #숫자 보기 싫으면 코드에서 한줄 삭제하장