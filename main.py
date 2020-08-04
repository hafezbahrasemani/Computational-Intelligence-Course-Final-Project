import file_handler
import random
import FCM
import numpy as np
if __name__ == '__main__':
    data = file_handler.read_file('input_data/2clstrain1200.csv')

    input_size = len(data)
    train_data = []
    test_data = []
    # producing random data for final drawing X:[-5, 17.5], Y:[-2.5, 15]
    X_rand, Y_rand = FCM.produce_random_num(-8, 19, -4, 17, 2000)

    random_input = np.concatenate((X_rand, Y_rand), axis=0)

    # Here we should choose randomly train and test data
    train_size = int(70 * input_size / 100)
    test_size = input_size - train_size

    train_data = [data[i] for i in range(test_size, input_size)]
    test_data = [data[i] for i in range(0, test_size)]

    train_data_no_label = []
    train_labels = []

    test_data_no_label = []
    test_labels = []

    for i in range(len(train_data)):
        train_data_no_label.append([train_data[i][0], train_data[i][1]])
        train_labels.append(train_data[i][2])

    for i in range(len(test_data)):
        test_data_no_label.append([test_data[i][0], test_data[i][1]])
        test_labels.append(test_data[i][2])

    # print(train_data_no_label)
    # print(train_data)
    # print(test_data)
    C = []
    U = []
    g = []

    m = 2.5
    max_iteration = 1000
    num_of_clusters = 2
    gama = 0.2

    num_of_classes = 2

    gama += 0.3
    cov, C, U, g = FCM.calculate_g_mat(train_data_no_label,
                                       num_of_clusters, m, max_iteration, gama)

    U_space = FCM.calculate_U(random_input, C, num_of_clusters, m, max_iteration)

    Y = FCM.calculate_Y(train_labels, num_of_classes)
    W, y = FCM.calculate_W(g, Y)

    # print(g)
    Y_test = FCM.calculate_Y(test_labels, num_of_classes)
    G_prime = FCM.calculate_g_mat_prime(test_data_no_label, C, U, cov, gama)

    y_test = FCM.calculate_y_test(G_prime, W)

    # print(U_space)

    # print(y)

    # print("Train Accuracy = " + str(FCM.accuracy(Y, y)) + "%" +
    #       " Number of clusters: " + str(num_of_clusters))
    print("Test Accuracy = " + str(FCM.accuracy(Y_test, y_test)) + "%" +
          " Number of clusters: " + str(num_of_clusters) + "Gama: " + str(gama))

    # print(len(train_data_no_label), len(U))
    FCM.draw_data(U, U_space, train_data_no_label, C, random_input, gama, max_iteration)
