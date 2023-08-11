import gtls
from GammaGWR import GammaGWR

if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network
    import_flag = False
    # Train AGWR with imported dataset    
    train_flag = True
    # Compute classification accuracy    
    test_flag = True
    # Export pickled network     
    export_flag = False
    # Export result data
    result_flag = True    
    # Plot network (2D projection)
    plot_flag = True
    
    if data_flag:
        ds_iris = gtls.Dataset(file='C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\4공정\\CSV\\4공정_FRONT_CYCLE.mp4pose_world_interpolated_visibility제거_하반신제거_레이블_첫행제거_레이블값1뺐음.csv', normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    if import_flag:
        fname = 'my_net.ggwr'
        my_net = gtls.import_network(fname, GammaGWR)

    if train_flag:
       # Create network 
       my_net = GammaGWR()
       # Initialize network with two neurons
       my_net.init_network(ds=ds_iris, random=False, num_context=1)
       # Train network on dataset
       my_net.train_ggwr(ds=ds_iris, epochs=15, a_threshold=0.35, beta=0.7, l_rates=[0.2, 0.001])
       
    if test_flag:
        my_net.test_gammagwr(ds_iris, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)
 
    if result_flag:
        fname= 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\결과\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨_첫행feature없음_결과.csv'
        gtls.export_result(fname, my_net, ds_iris)

    if export_flag:
        fname = 'my_net.ggwr'
        gtls.export_network(fname, my_net)

    if plot_flag:
        gtls.plot_network_hs(my_net, edges=True, labels=True)