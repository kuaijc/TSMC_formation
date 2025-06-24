import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment


# randomly generate the undirected connected communication graph
def generate_connected_adjacency_matrix():
    connected = False
    while not connected:
        graph = nx.erdos_renyi_graph(N, p)
        connected = nx.is_connected(graph)
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    return np.array(adjacency_matrix, dtype=int)


# centralized Hungarian algorithm used to determine the optimal match relationship
def hungarian(cost_f):
    row, col = linear_sum_assignment(cost_f)
    result = np.zeros((N, N))
    for e in range(N):
        result[e][col[e]] = 1
    return result


# constraint function g(.)
def g_function(position, iii):
    vector_g = np.array([1, -2] + [0 for _ in range(N - 2)])
    return np.dot(vector_g.T, position)


# used to calculate a gradient
def calculate_gradient(f, x_k, iii):
    h = 1e-5
    x_k = np.array([float(xx) for xx in x_k])
    grad = []
    for jj in range(len(x_k.tolist())):
        x_k_plus_h = x_k.copy()
        x_k_plus_h[jj] += h
        gradient_i = (f(x_k_plus_h, iii) - f(x_k, iii)) / h
        grad.append(gradient_i)
    grad = np.array(grad)
    return grad


# objective function of P1
def location_function(do, i_):
    d0 = np.array(do)
    answer = np.linalg.norm(location_q[i_] - matched_location_m[i_] - d0) ** 2
    return answer


# used to calculate the relative error in optimal match part
def calculate_matching_error(xx):
    x_star = optimal_assignment
    delta_x = xx - x_star
    abs_error = np.linalg.norm(delta_x, ord=2)
    rel_error = abs_error / np.linalg.norm(x_star, ord=2)
    if rel_error == 0:
        rel_error = 1e-16   # avoid the math error of 0 in log
    return rel_error


np.random.seed(42)
random.seed(42)
if __name__ == '__main__':
    N = 34  # agent num
    p = 0.5  # edge generation probability
    Adjacency_Matrix = generate_connected_adjacency_matrix()   # randomly generate communication graph
    Ni = np.sum(Adjacency_Matrix, axis=1).astype(int)   # Neighborhood set

    # parameter selection
    alpha = 0.14    # the tunable step-size of DC-ADMM
    rho = 0.05  # the tunable penalty force of DC-ADMM
    beta = 0.025     # the tunable step-size of D-OGDA
    D = 0.5     # the tunable dual variable bound of D-OGDA

    matching_times = 5000   # match-iteration times
    location_times = 700    # location-iteration times

    IN = np.ones(N, dtype=float)
    ON = np.zeros(N, dtype=float)
    IIN = np.identity(N)

    # randomly generate the initial positions of the agents
    location_q = np.array(
        [[round(random.uniform(-2, 3), 2) for _ in range(2)] + [0 for _ in range(N - 2)] for _ in range(N)])

    # give the desired formation shape as a 'BIT'
    sq_3 = math.sqrt(3)
    relation_m = np.array([[0, 4], [1, 4], [-1, 4], [0, 2], [0, 0], [0, -2], [0, -4], [1, -4], [-1, -4],
                           [6, 4], [5, 4], [4, 4], [7, 4], [8, 4], [6, 2], [6, 0], [6, -2], [6, -4],
                           [-6.5, 4], [-6.5, 2.4], [-6.5, 0.8], [-6.5, -0.8], [-6.5, -2.4], [-6.5, -4], [-4.5, 2],
                           [-4.5, -2],
                           [-5.5, 2 + sq_3], [-5.5, 2 - sq_3], [-5.5, -2 + sq_3], [-5.5, -2 - sq_3],
                           [-6.5 + sq_3, 3], [-6.5 + sq_3, 1], [-6.5 + sq_3, -1], [-6.5 + sq_3, -3]])
    relation_m = np.array([0.1 * xx for xx in relation_m])
    location_relation_m = np.hstack((relation_m, np.zeros((relation_m.shape[0], N - 2), dtype=int)))    # dimension
    arbitrary_d0 = np.array(
        [random.uniform(0, 1) for _ in range(2)] + [0 for _ in range(N - 2)])      # arbitrarily generate a center
    arbitrary_m = location_relation_m + arbitrary_d0        # position of hole under arbitrary center
    cost = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance = np.array(arbitrary_m[j] - location_q[i])
            cost[i][j] = np.dot(distance.T, distance)   # calculate the cost
    optimal_assignment = hungarian(cost)

    # the first problem: find the optimal matching relationship
    ini_x = [[round(random.uniform(0, 1), 2) for _ in range(N)] for _ in range(N)]
    ini_y = [[round(random.uniform(0, 1), 2) for _ in range(N)] for _ in range(N)]
    ini_lam = [round(random.uniform(0, 1), 2) for _ in range(N)]
    ini_psi = np.zeros((N, N))  # iteration variable initialization

    x = np.array(ini_x.copy())
    y = np.array(ini_y.copy())
    lam = np.array(ini_lam.copy())
    psi = np.array(ini_psi.copy())
    matching_error = []
    f_value = []

    k = 0
    while k < matching_times:
        matching_error.append(calculate_matching_error(x))      # calculate relative error
        if k > 200:
            if all(x < 1e-8 for x in matching_error[-200:]):    # ensure the stability
                break

        x_copy = x.copy()
        y_copy = y.copy()
        lam_copy = lam.copy()
        psi_copy = psi.copy()

        for i in range(N):
            sigma_y = np.zeros(N, dtype=float)
            for j in range(N):
                if Adjacency_Matrix[i][j] == 1:
                    sigma_y += y_copy[i] + y_copy[j]
            w_xi = (IN / N - x_copy[i] - psi_copy[i] + rho * sigma_y) / (2 * rho * Ni[i])
            xi_hat = x_copy[i] - alpha * (cost[i] + (lam_copy[i] + np.dot(IN.T, x_copy[i]) - 1) * IN - w_xi)    # 18a
            x[i] = np.array([min(1.0, max(0.0, xi_hat[j])) for j in range(N)])  # equation 18b
            y[i] = (IN / N - x[i] - psi_copy[i] + rho * sigma_y) / (2 * rho * Ni[i])  # equation 16
            lam[i] = lam_copy[i] + alpha * (np.dot(IN.T, x[i]) - 1)     # equation 18c

        for i in range(N):
            minus_y = np.zeros(N, dtype=float)
            for j in range(N):
                if Adjacency_Matrix[i][j] == 1:
                    minus_y += y[i] - y[j]
            psi[i] = psi_copy[i] + rho * minus_y    # equation 13

        last_x = x_copy.copy()
        last_lam = lam_copy.copy()
        k += 1
        print('match part--iteration times：', k, '/', matching_times)

    # print('calculated task assignment：\n', x)
    # print('accuracy：', matching_error[-1])
    match_x_amount = len(matching_error)
    match_x = np.arange(match_x_amount)
    
    # # error visualization
    # plt.figure(figsize=(8, 6))
    # plt.yscale('log')
    # plt.plot(match_x, matching_error, linewidth=2, label='matching_error')
    # plt.xlabel('Iteration Times k', fontsize=18, fontname='Times New Roman')
    # plt.ylabel('Relative Error', fontsize=18, fontname='Times New Roman')
    # plt.xlim(0, match_x_amount)
    # plt.show()

    # calculate the hole position with right match relationship
    matched_location_m = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if x[i][j] == 1:
                matched_location_m[i] = location_relation_m[j]

    # the second problem: find optimal reference center d0*
    d_0 = np.array([[round(random.uniform(0, 1), 2) for _ in range(N)] for _ in range(N)])
    w_0 = np.array([[round(random.uniform(0, 1), 2) for _ in range(N)] for _ in range(N)])
    miu_0 = np.array([round(random.uniform(0, 1), 2) for _ in range(N)])
    d_1 = d_0.copy()
    w_1 = w_0.copy()
    miu_1 = miu_0.copy()
    d = d_0.copy()
    last_d = d_1.copy()
    w = w_0.copy()
    last_w = w_1.copy()
    miu = miu_0.copy()
    last_miu = miu_1.copy()     # iteration variable initialization
    d_sequence = []

    k2 = 0
    while k2 < location_times:
        d_copy = np.array(d.copy())
        last_d_copy = np.array(last_d.copy())
        w_copy = np.array(w.copy())
        last_w_copy = np.array(last_w.copy())
        miu_copy = np.array(miu.copy())
        last_miu_copy = np.array(last_miu.copy())
        d_sequence.append(d_copy)

        for i in range(N):
            sigma_k = np.zeros(N, dtype=float)
            sigma_last_k = np.zeros(N, dtype=float)
            minus_k = np.zeros(N, dtype=float)
            minus_last_k = np.zeros(N, dtype=float)
            for j in range(N):
                if Adjacency_Matrix[i][j] == 1:
                    sigma_k += w_copy[i] - w_copy[j] + d_copy[i] - d_copy[j]
                    sigma_last_k += last_w_copy[i] - last_w_copy[j] + last_d_copy[i] - last_d_copy[j]
                    minus_k += d_copy[i] - d_copy[j]
                    minus_last_k += last_d_copy[i] - last_d_copy[j]

            d[i] = d_copy[i] - 2 * beta * (calculate_gradient(location_function, d_copy[i], i) + sigma_k    # 21a
                + miu_copy[i] * calculate_gradient(g_function, (d_copy[i] + location_relation_m[i]), i)) \
                + beta * (calculate_gradient(location_function, last_d_copy[i], i) + sigma_last_k
                + last_miu_copy[i] * calculate_gradient(g_function, (last_d_copy[i] + location_relation_m[i]), i))
            w[i] = w_copy[i] + 2 * beta * minus_k - beta * minus_last_k     # equation 21b
            scalar_miu = miu_copy[i] + 2 * beta * g_function(d_copy[i] + location_relation_m[i], i) \
                        - beta * g_function(last_d_copy[i] + location_relation_m[i], i)
            miu[i] = min(D, max(0, scalar_miu))     # equation 21c
        last_d = d_copy.copy()
        last_w = w_copy.copy()
        last_miu = miu_copy.copy()
        k2 += 1
        print('location part--iteration times：', k2, '/', location_times)

    locating_error = []
    for i in range(len(d_sequence)):    # calculate the relative error in optimal reference center part
        delta_d = d_sequence[i] - d
        absolute_error = np.linalg.norm(delta_d, ord=2)
        relative_error = absolute_error / np.linalg.norm(d, ord=2)
        if relative_error == 0:
            relative_error = 1e-16
        locating_error.append(relative_error)
    # print('accuracy:', locating_error[-1])
    location_x_amount = len(locating_error)
    locate_x = np.arange(location_x_amount)

    # # error visualization
    # plt.figure(figsize=(8, 6))
    # plt.yscale('log')
    # plt.plot(locate_x, locating_error, linewidth=2, label='location_error')
    # plt.xlabel('Iteration Times k', fontsize=18, fontname='Times New Roman')
    # plt.ylabel('Relative Error', fontsize=18, fontname='Times New Roman')
    # plt.xlim(0, location_times)
    # plt.show()

    task_position = []
    for i in range(N):
        task_position.append(d[i] + matched_location_m[i])     # the position of real hole with known center
    robot_position = location_q.copy()
    robot_2d = [(point[0], point[1]) for point in robot_position]
    task_2d = [(point[0], point[1]) for point in task_position]
    x_values1 = [point[0] for point in robot_2d]
    y_values1 = [point[1] for point in robot_2d]
    x_values2 = [point[0] for point in task_2d]
    y_values2 = [point[1] for point in task_2d]

    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values1, y_values1, color='red', label='Agents')
    plt.scatter(x_values2, y_values2, color='green', label='Desired formation configuration')
    for i in range(len(x_values1)):
        if i == 0:
            plt.plot([x_values1[i], x_values2[i]], [y_values1[i], y_values2[i]], 'k--',
                     linewidth=2, label='Optimal matching relationship')
        else:
            plt.plot([x_values1[i], x_values2[i]], [y_values1[i], y_values2[i]], 'k--', linewidth=2)
    plt.xlabel('X Axis', fontsize=20, fontname='Times New Roman')
    plt.ylabel('Y Axis', fontsize=20, fontname='Times New Roman')
    plt.xlim(-2.5, 5)
    plt.ylim(-2.5, 5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'family': 'Times New Roman', 'size': 16}, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    # plt.savefig(r'D:\学校\硕士\大四下\编队匹配论文\第一次大修\picture\formation_2.pdf', format='pdf')
    plt.show()

    # visualization the raltive error
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 4))

    # match error is placed in the left
    plt.subplot(1, 2, 1)
    plt.yscale('log')
    plt.plot(match_x, matching_error, color='blue', linewidth=2, label='match_error')
    plt.xlabel('Iteration Times k', fontsize=20, fontname='Times New Roman')
    plt.ylabel(r'$\frac{\Vert \chi-\chi^{*}\Vert_{2}}{\Vert \chi^{*}\Vert_{2}}$',
               fontsize=20, fontname='Times New Roman')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, match_x_amount)
    plt.legend(prop={'family': 'Times New Roman', 'size': 15}, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # location error is placed in the right
    plt.subplot(1, 2, 2)
    plt.yscale('log')
    plt.plot(locate_x, locating_error, color='red', linewidth=2, label='location_error')
    plt.xlabel('Iteration Times k', fontsize=20, fontname='Times New Roman')
    plt.ylabel(r'$\frac{\Vert d_{0}-d_{0}^{*}\Vert_{2}}{\Vert d_{0}^{*}\Vert_{2}}$',
               fontsize=20, fontname='Times New Roman')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, location_times)
    plt.legend(prop={'family': 'Times New Roman', 'size': 15}, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    # plt.savefig(r'D:\学校\硕士\大四下\编队匹配论文\第一次大修\picture\error_2.pdf', format='pdf')
    plt.show()

