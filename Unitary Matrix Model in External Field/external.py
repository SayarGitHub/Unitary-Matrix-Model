import numpy as np
from numpy.linalg import qr
from numpy.linalg import eig
import random
import math
from mpmath import mp
import pandas as pd


def gen_action(U, A):
    temp = np.dot(U, np.conj(A.T)) + np.dot(A, np.conj(U.T))
    t = np.trace(temp)
    return -math.sqrt((t * np.conj(t)).real)


def monte_carlo(
    U,
    A,
    iterations,
    Lambda,
    is_therm,
    N,
    EPS,
    GAP,
    a_rate,
    accept,
    no_calls,
    action_therm,
    action,
):
    n = 0
    result = [None] * 3
    p_avg = 0
    p_sq = 0
    p_err = 0

    for i in range(iterations):
        no_calls += 1
        old_action = gen_action(U, A)

        U_new = np.copy(U)
        # for j in range(N):
        #     for k in range(N):

        index_a = np.random.randint(0, N)
        index_b = np.random.randint(0, N)
        noise = 2 * (random.random() - 0.5) + 2 * (random.random() - 0.5) * 1j
        U_new[index_a][index_b] += EPS * noise

        U_new, _ = qr(U_new, mode="complete")
        new_action = gen_action(U_new, A)

        delta_S = new_action - old_action
        u = random.random()

        if mp.exp(-delta_S) >= u or delta_S < 0:
            accept += 1
            U = np.copy(U_new)
            if is_therm == 0:
                action.append(new_action)
            else:
                action_therm.append(new_action)
        else:
            if is_therm == 0:
                action.append(old_action)
            else:
                action_therm.append(old_action)

        if (is_therm == 0) and (no_calls % 100 == 0) and (no_calls > 1):
            a_rate.append(accept * 100.0 / no_calls)

        if (is_therm == 0) and (i % GAP == 0):
            n += 1
            # t = np.trace(np.dot(np.dot(U, U), U))
            t = np.trace(np.dot(np.dot(np.dot(U, U), U), U))
            if Lambda < 2:
                # r = math.sqrt((t * np.conj(t)).real) / N
                r = t.real / N
            else:
                r = t.real / N
            p_avg += r
            p_sq += r * r

    if is_therm == 0:
        p_avg /= n
        p_sq /= n
        p_err = math.sqrt(abs(p_sq - p_avg * p_avg) / n)
        result[0] = p_avg
        result[1] = p_err

        if Lambda < 2:
            result[2] = ((1 - (Lambda / 2)) ** 2) * (
                1 - 3 * Lambda + (7 * (Lambda ** 2) / 4)
            )
            # result[2] = -((1 - (Lambda / 2)) ** 2) * (1 - (5 * Lambda / 4))
        else:
            result[2] = 0

    return result, U, a_rate, accept, no_calls, action_therm, action


# def get_x(eig, N, s):
#     if s <= 2:
#         return 0
#     else:
#         # return (1 / 4) - (sum(eig) / N)
#         x = 0
#         sum = 0
#         for i in eig:
#             sum += 1 / math.sqrt(i + x)
#         sum /= N
#         d = sum - 2
#         while d > 0.005 or d < -0.005:

#             if d > 0:
#                 x += d / 10
#             elif d < 0:
#                 x -= d / 10
#             else:
#                 break

#             sum = 0
#             for i in eig:
#                 sum += 1 / math.sqrt(i + x)
#             sum /= N
#             d = sum - 2

#         return x


# def free_energy(eig, N, x):
#     A = 0
#     B = 0
#     C = 0
#     for i in eig:
#         A += (2 / N) * np.sqrt(i + x)
#         for j in eig:
#             B += (1 / 2 * N) * (np.sqrt(i + x) + np.sqrt(j + x))
#     C = x + (3 / 4)
#     return A - B - C


def main():

    N = int(input("Please enter the size of the matrix: "))
    EPS = float(input("Please enter the step size: "))
    GAP = int(input("Please enter the measurement interval: "))

    therm = 20000
    sweeps = 100000
    total_iter = 20

    lambda_list = []
    p_avg_list = []
    p_err_list = []
    p_expec_list = []
    s_list = []
    # x_list = []
    # F_list = []
    rate_list = []
    therm_list = []
    act_list = []

    eigen_list = []
    Lambda = 0.01
    f = 0.01

    # A = (np.eye(N, N) + np.diag(np.diag(np.random.rand(N, N)))) * (N / f)

    for i in range(total_iter):
        accept = 0
        no_calls = 0
        a_rate = []
        action_therm = []
        action = []

        A = np.eye(N) * (N / Lambda)
        # A = (np.eye(N, N) + np.diag(np.diag(np.random.rand(N, N)))) * (N / f)
        e_val, _ = eig(np.dot(np.conj(A.T), A))

        s = 0
        for j in e_val:
            s += 1 / math.sqrt(j)

        eigen_list.append(e_val)
        # x = get_x(e_val, N, s)

        # Lambda = 0.01
        # for j in range(total_iter):
        #     accept = 0
        #     no_calls = 0
        #     a_rate = []

        U = np.eye(N) + 0.01 * (np.random.rand(N, N) + np.random.rand(N, N) * 1j)
        U, _ = qr(U, mode="complete")

        _, U, _, _, _, action_therm, _ = monte_carlo(
            U,
            A,
            therm,
            Lambda,
            1,
            N,
            EPS,
            GAP,
            a_rate,
            accept,
            no_calls,
            action_therm,
            action,
        )

        result, U, a_rate, accept, no_calls, _, action = monte_carlo(
            U,
            A,
            sweeps,
            Lambda,
            0,
            N,
            EPS,
            GAP,
            a_rate,
            accept,
            no_calls,
            action_therm,
            action,
        )

        lambda_list.append(Lambda)
        s_list.append(s)
        p_avg_list.append(result[0])
        p_err_list.append(result[1])
        p_expec_list.append(result[2])
        # x_list.append(x)
        # F_list.append(free_energy(e_val, N, x))
        rate_list.append(a_rate)
        therm_list.append(action_therm)
        act_list.append(action)
        Lambda += 0.25

        if Lambda > 1 and Lambda < 2:
            EPS = 0.093
        elif Lambda > 2 and Lambda < 3:
            EPS = 0.093
        elif Lambda > 3 and Lambda < 4:
            EPS = 1.693
        elif Lambda > 4:
            EPS = 1.893
        # EPS += 0.045
        # print(((i * total_iter) + j) * 100 / (total_iter * total_iter))
        print(i * 100 / total_iter)

        f += 0.25

    df = pd.DataFrame(eigen_list, columns=range(N))
    df.to_csv("eigenvalues.csv", index=False)

    df = pd.DataFrame(rate_list, columns=range(len(rate_list[0])))
    df.to_csv("Acceptance_rate.csv", index=False)

    df = pd.DataFrame(therm_list, columns=range(therm))
    df.to_csv("Thermalization.csv", index=False)

    df = pd.DataFrame(act_list, columns=range(sweeps))
    df.to_csv("Action.csv", index=False)

    df_list = []
    for i in range(len(lambda_list)):
        df_list.append(
            [
                lambda_list[i],
                s_list[i],
                p_avg_list[i],
                p_err_list[i],
                p_expec_list[i]
                # x_list[i],
                # F_list[i],
            ]
        )
    df = pd.DataFrame(df_list, columns=["Lambda", "S", "P", "Error", "Expected"])
    df.to_csv("Data.csv", index=False)


if __name__ == "__main__":
    main()
