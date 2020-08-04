import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

global maximum
maximum = 10000.0

global Threshold
Threshold = 0.000000001


def draw_data(U, U_space, train_data, Centers, random_input, m, max_iteration):
    colors = ['blue', 'green', 'yellow', 'brown', 'pink', 'magenta', 'C0', 'C2', 'C4', 'C6']

    Centers_x = []
    Centers_y = []

    train_data_x = []
    train_data_y = []

    random_input_x = []
    random_input_y = []

    for i in range(len(Centers)):
        Centers_x.append(Centers[i][0])
        Centers_y.append(Centers[i][1])

    for i in range(len(random_input)):
        random_input_x.append(random_input[i][0])
        random_input_y.append(random_input[i][1])

    for i in range(len(train_data)):
        train_data_x.append(train_data[i][0])
        train_data_y.append(train_data[i][1])


    for i in range(len(U_space)):
        plt.plot(random_input_x[i], random_input_y[i], 'bo', color=colors[np.argmax(U_space[i])])


    for i in range(len(U)):
        plt.plot(train_data_x[i], train_data_y[i], 'bo', color=colors[np.argmax(U[i])])

    # plt.plot(train_data_x, train_data_y, 'bo', color='blue')
    plt.xlabel("Gama = " + str(m) +
               "   max_iteration = " + str(max_iteration))

    # plt.plot(X_test, y_test, 'bo', color='blue')
    plt.plot(Centers_x, Centers_y, 'ro', color='black')

    plt.show()


def distance_from_center(point, center):
    temp = 0.0

    for i in range(0, len(point)):
        temp += abs(point[i] - center[i]) ** 2

    return math.sqrt(temp)


def terminate_condition(U, U_old):
    global Threshold

    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Threshold:
                return False
    return True


def initialize_membership(input_data, num_of_clusters):
    global maximum
    U = []
    for i in range(0, len(input_data)):
        current = []
        row_sum = 0.0

        for j in range(0, num_of_clusters):
            temp = random.randint(1, int(maximum))
            current.append(temp)
            row_sum += temp

        # We ensure that their sum is equal to 1
        for j in range(0, num_of_clusters):
            current[j] = current[j] / row_sum

        U.append(current)
    return U


def calculate_center_vectors(input_data, num_of_clusters, m, max_iteration):
    U = initialize_membership(input_data, num_of_clusters)

    iterate_num = 0
    # Loop to calculate membership and centers until the end condition is true
    while True:
        iterate_num += 1

        # Save a copy of all elements of U in U_old
        U_old = copy.deepcopy(U)

        # clusters centers vectors
        Centers = []

        for i in range(0, num_of_clusters):
            curr_center = []

            for j in range(0, len(input_data[0])):
                dividend_sum = 0.0
                divisor_sum = 0.0

                for k in range(0, len(input_data)):
                    dividend_sum += pow(U[k][i], m) * input_data[k][j]
                    divisor_sum += pow(U[k][i], m)
                curr_center.append(dividend_sum / divisor_sum)
            Centers.append(curr_center)

        # Update memberships

        # 1- Creating a distance matrix which d[i][j] is the distance from X[i] to C[j]
        d_matrix = []
        for i in range(0, len(input_data)):
            curr = []
            for j in range(0, num_of_clusters):
                curr.append(distance_from_center(input_data[i], Centers[j]))
            d_matrix.append(curr)

        # 2- Update U
        for j in range(0, num_of_clusters):
            for i in range(0, len(input_data)):
                sum = 0.0
                for k in range(0, num_of_clusters):
                    sum += pow(d_matrix[i][j] / d_matrix[i][k], 2 / (m - 1))
                U[i][j] = 1 / sum

        # 3- Check terminate condition
        if terminate_condition(U, U_old):
            print("Clustering is done by difference condition"
                  + " after " + str(iterate_num) + " iterations")
            break

        if iterate_num > max_iteration:
            print("Clustering is done by max_iteration condition")
            break

    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1

    return Centers, U


def calculate_U(input_data, Centers, num_of_clusters, m, max_iteration):
    U = initialize_membership(input_data, num_of_clusters)

    iterate_num = 0
    # Loop to calculate membership and centers until the end condition is true
    while True:
        iterate_num += 1

        # Save a copy of all elements of U in U_old
        U_old = copy.deepcopy(U)

        # Update memberships

        # 1- Creating a distance matrix which d[i][j] is the distance from X[i] to C[j]
        d_matrix = []
        for i in range(0, len(input_data)):
            curr = []
            for j in range(0, num_of_clusters):
                curr.append(distance_from_center(input_data[i], Centers[j]))
            d_matrix.append(curr)

        # 2- Update U
        for j in range(0, num_of_clusters):
            for i in range(0, len(input_data)):
                sum = 0.0
                for k in range(0, num_of_clusters):
                    sum += pow(d_matrix[i][j] / d_matrix[i][k], 2 / (m - 1))
                U[i][j] = 1 / sum

        # 3- Check terminate condition
        if terminate_condition(U, U_old):
            # print("Clustering is done by difference condition"
            #       + " after " + str(iterate_num) + " iterations")
            break

        if iterate_num > max_iteration:
            print("Clustering is done by max_iteration condition")
            break

    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1

    return U


def calculate_covariance_mat(data, centers, U, m):
    C = []
    centers = np.array(centers)

    for i in range(0, len(centers)):


        shape = (2, 2)
        dividend_sum = np.zeros(shape)
        divisor_sum = 0.0

        for k in range(0, len(data)):
            dividend_sum += pow(U[k][i], m) * np.multiply(
                np.array([data[k] - centers[i]]), np.transpose(np.array([data[k] - centers[i]])))

            divisor_sum += pow(U[k][i], m)

        # print(dividend_sum)
        # print(curr_cov)
        C.append((1 / divisor_sum) * dividend_sum)

    return C


def calculate_g_mat(data, num_of_clusters, m, max_iteration, gama):
    g = np.zeros(shape=(num_of_clusters, len(data)))
    data = np.array(data)

    centers, U = calculate_center_vectors(data, num_of_clusters, m, max_iteration)
    covariance_mat = calculate_covariance_mat(data, centers, U, m)

    centers = np.array(centers)
    for i in range(0, len(centers)):
        for k in range(0, len(data)):
            g[i][k] = math.exp(-1 * gama * np.dot(np.dot(
                np.array([data[k] - centers[i]]), inv(covariance_mat[i])),
                np.transpose(np.array([data[k] - centers[i]]))))

    return covariance_mat, centers, U, g


def calculate_g_mat_prime(data, centers, U, covariance_mat, gama):
    g_prime = np.zeros(shape=(len(centers), len(data)))

    for i in range(0, len(centers)):
        for k in range(0, len(data)):
            g_prime[i][k] = math.exp(-1 * gama * np.dot(np.dot(
                np.array([data[k] - centers[i]]), inv(covariance_mat[i])),
                np.transpose(np.array([data[k] - centers[i]]))))

    return g_prime


def calculate_Y(labels, num_of_classes):
    y = np.zeros(shape=(len(labels), num_of_classes))

    for i in range(0, len(labels)):
        for j in range(1, num_of_classes + 1):
            if int(labels[i]) == j:
                y[i][j - 1] = 1
            else:
                y[i][j - 1] = 0
    return y


def calculate_W(G, Y):
    W = np.zeros(shape=(G.shape[1], Y.shape[1]))

    W = np.dot(np.dot(inv(np.dot(G, G.T)), G), Y)

    y = np.argmax(np.dot(G.T, W), axis=1)
    return W, y


def calculate_y_test(G_prime, W):
    y = np.argmax(np.dot(G_prime.T, W), axis=1)
    return y


def accuracy(y, y_bar):
    sum = 0.0

    for i in range(0, len(y)):
        temp = np.argmax(y[i], axis=0)
        sum += abs(temp - y_bar[i])

    return (1.0 - (sum / len(y))) * 100


def sign(x):
    result = 0
    if x > 0:
        result = 1
    elif x < 0:
        result = -1
    else:
        result = 0

    return result


def produce_random_num(x_min, x_max, y_min, y_max, num_of_points):
    X_rand = np.random.uniform(x_min, x_max, size=(num_of_points, 2))
    Y_rand = X_rand = np.random.uniform(y_min, y_max, size=(num_of_points, 2))

    return X_rand, Y_rand

