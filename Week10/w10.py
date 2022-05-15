import math
import numpy as np
from itertools import chain
from commons.utils import distance
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from sklearn.cluster import AgglomerativeClustering


def Kmeans(S=[[-1, 1, 0, 4, 3, 5], [3, 4, 5, -1, 0, 1]], m1=[[-1], [3]], m2=[[5], [1]]):
    iteration = 1
    while 1:
        print("iteration:" + str(iteration))
        print("SampleX  \t ||x-m1||  \t ||x-m2||  \t  Class")
        class1 = []
        class2 = []
        for i in range(len(S[0])):
            dist1 = math.sqrt((m1[0][0] - S[0][i]) * (m1[0][0] - S[0][i]) + (m1[1][0] - S[1][i]) * (m1[1][0] - S[1][i]))
            dist2 = math.sqrt((m2[0][0] - S[0][i]) * (m2[0][0] - S[0][i]) + (m2[1][0] - S[1][i]) * (m2[1][0] - S[1][i]))
            if (dist1 <= dist2):
                class1.append([S[0][i], S[1][i]])
                label = 1
            else:
                class2.append([S[0][i], S[1][i]])
                label = 2
            print("[" + str(S[0][i]) + " " + str(S[1][i]) + "]" + "   " + str(dist1) + "         " + str(
                dist2) + "         " + str(label))
            # print(f"[{S[0][i]}, {S[1][i]}] \t \t {round(dist1, 5)} \t \t {round(dist2, 5)} \t \t {label}")
        new_m1 = np.mean(class1, axis=0)
        new_m2 = np.mean(class2, axis=0)
        
        if (np.array_equal(list(np.array(m1).flatten()), new_m1) and np.array_equal(list(np.array(m2).flatten()),
                                                                                    new_m2)):
            break
        else:
            m1 = [[new_m1[0]], [new_m1[1]]]
            m2 = [[new_m2[0]], [new_m2[1]]]
        iteration += 1
    
    print("final m1:")
    print(m1)
    print("final m2:")
    print(m2)


def Kmeans_v2(S=[[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]],
              centers=[[-1, 3], [5, 1]],
              iterations=2):
    for iteration in range(iterations):
        print("--------------------------------------------")
        print(f"iteration {iteration}" )
        print(f"centers: {centers}")
        cls = dict()
        for i in range(len(S)):
            dst = []    # distances for each data sample to all centers
            for center in centers:
                dst.append(distance(S[i], center))
            belong = 1 + np.argmin(dst)
            if belong not in cls:
                cls[belong] = []
            cls[belong].append(S[i])
            # cls[i] = 1 + np.argmin(dst)
            print(f"Sample: {S[i]}", end="")
            for j in range(len(dst)):
                print(f", \t d to {centers[j]}: {round(dst[j], 4)}", end="")
            print(f", \t class {belong}")
        # Update centers
        for i in range(len(centers)):
            new_list = np.mean(cls[i + 1], axis=0).tolist()
            centers[i] = new_list
        print(f"New center: {centers}")
        
        
def kmeans_sklearn(S=[[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]],
                   n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters).fit(S)
    print("Centers: (converged)")
    print(kmeans.cluster_centers_)


def Fuzzy_kmeans(S=[[-1, 1, 0, 4, 3, 5], [3, 4, 5, -1, 0, 1]],
                 mu=[[1, 0.5, 0.5, 0.5, 0.5, 0], [0, 0.5, 0.5, 0.5, 0.5, 1]],
                 b=2, theta=0.5):
    iteration = 1
    while (1):
        print("iteration:" + str(iteration))
        print("normalise memberships:")
        print(mu)

        first_element = 0
        second_element = 0
        dinominator = 0
        for i in range(len(mu[0])):
            first_element += mu[0][i] * mu[0][i] * S[0][i]
            second_element += mu[0][i] * mu[0][i] * S[1][i]
            dinominator += mu[0][i] * mu[0][i]
        m1 = [[first_element / dinominator], [second_element / dinominator]]

        first_element = 0
        second_element = 0
        dinominator = 0
        for i in range(len(mu[0])):
            first_element += mu[1][i] * mu[1][i] * S[0][i]
            second_element += mu[1][i] * mu[1][i] * S[1][i]
            dinominator += mu[1][i] * mu[1][i]
        m2 = [[first_element / dinominator], [second_element / dinominator]]

        print("m1 = " + str(m1))
        print("m2 = " + str(m2))

        new_mu1 = []
        new_mu2 = []
        print(
            "SampleX   ||x-m1||      ||x-m2||         (1/||x-m1||)^(2/(b-1))       (1/||x-m2||)^(2/(b-1))       u1j               u2j")
        for i in range(len(S[0])):
            dist1 = math.sqrt((m1[0][0] - S[0][i]) * (m1[0][0] - S[0][i]) + (m1[1][0] - S[1][i]) * (m1[1][0] - S[1][i]))
            dist2 = math.sqrt((m2[0][0] - S[0][i]) * (m2[0][0] - S[0][i]) + (m2[1][0] - S[1][i]) * (m2[1][0] - S[1][i]))

            u1j = 1 / (dist1 * dist1) / (1 / (dist1 * dist1) + 1 / (dist2 * dist2))
            u2j = 1 / (dist2 * dist2) / (1 / (dist1 * dist1) + 1 / (dist2 * dist2))

            print("[" + str(S[0][i]) + " " + str(S[1][i]) + "]" + "   " + str(round(dist1, 4)) + "          " + str(
                round(dist2, 4)) + "                  " + str(
                round(1 / pow(dist1, 2 / (b - 1)), 4)) + "                          " + str(
                round(1 / pow(dist2, 2 / (b - 1)), 4)) + "           "
                  + str(round(u1j, 4)) + "           " + str(round(u2j, 4)))

            new_mu1.append(u1j)
            new_mu2.append(u2j)

        new_mu = [new_mu1, new_mu2]
        print()
        if iteration > 1:
            if (abs(m1[0][0] - m1_old[0][0]) < theta and abs(m1[1][0] - m1_old[1][0]) < theta and
                    abs(m2[0][0] - m2_old[0][0]) < theta and abs(m2[1][0] - m2_old[1][0]) < theta):
                print("final m1:")
                print(m1)
                print("final m2:")
                print(m2)
                break

        mu = new_mu

        m1_old = m1
        m2_old = m2
        iteration += 1

def competitive_learning(C=[[-0.5, 1.5], [0, 2.5], [1.5, 0]],
                         S=[[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]],
                         chosen_id=[3, 1, 1, 5, 6],
                         lr=0.1):
    # Initialize Table
    table = PrettyTable()
    columns_name = ["Iteration", "X"]
    for i in range(len(C)):
        columns_name.append(f"||x-m{i+1}||")
    columns_name.append("j")
    columns_name.append("mj<-mj+lr*(x-mj)")
    table.field_names = columns_name

    distances = np.array([0]*len(C), dtype=np.float)
    C = np.array(C)
    ite = 0
    for id in chosen_id:
        table_rows = []
        ite += 1
        table_rows.append(str(ite))
        x = S[id-1]
        table_rows.append(str(x))
        x = np.array(x)
        for i in range(len(C)):
            distances[i] = np.around(distance(x1=x, x2=C[i]), 4)
            table_rows.append(str(distances[i]))
        selected_j = np.argmin(distances)
        table_rows.append(str(selected_j+1))
        C[selected_j] += lr * (x - C[selected_j])
        table_rows.append(str(C[selected_j]))
        table.add_row(table_rows)
    print(table)

def leader_follower_clustering(S=[[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]],
                            chosen_id=[3, 1, 1, 5, 6],
                            theta = 3,
                            lr = 0.5):
    # Initialize Table
    table = PrettyTable()
    columns_name = ["Iteration", "X", "clustering centers", "||x-m_j||", "j", "||x-m_j||<theta", "update"]
    table.field_names = columns_name

    ite = 0
    cluster_centers = []
    for id in chosen_id:
        table_rows = []
        ite += 1
        table_rows.append(str(ite))
        x = S[id-1]
        table_rows.append(str(x))
        if(ite == 1):
            cluster_centers.append(x)
        x = np.array(x)
        table_rows.append(str(cluster_centers))
        distances = []
        for i in range(len(cluster_centers)):
            dist = np.around(distance(x1=x, x2=cluster_centers[i]), 4)
            distances.append(dist)
        table_rows.append(distances)
        selected_j = np.argmin(distances)
        table_rows.append(str(selected_j + 1))

        if(min(distances) < theta):
            table_rows.append("yes")
            cluster_centers[selected_j] += lr * (x - cluster_centers[selected_j])
        else:
            table_rows.append("no")
            cluster_centers.append(S[id-1])

        table_rows.append(str(cluster_centers))
        table.add_row(table_rows)
    print(table)


def Agglomerative(X=[[-1, 3], [1, 2], [0, 1], [4, 0], [5, 4], [3, 2]],
                  affinity='euclidean',
                  linkage='single'):
    table = PrettyTable()
    columns_name = ["Iteration"]
    for i in range(len(X)):
        columns_name.append(f"{X[i]}")
    table.field_names = columns_name
    for i in range(len(X), 0, -1):
        table_rows = [str(len(X) - i)]
        clustering = AgglomerativeClustering(n_clusters=i, affinity=affinity, linkage=linkage).fit(X)
        table_rows += clustering.labels_.tolist()
        table.add_row(table_rows)
    print(table)
    
    
if __name__ == '__main__':
    # Kmeans()
    # Fuzzy_kmeans()
    # Kmeans_v2()
    #kmeans_sklearn()
    # competitive_learning()
    leader_follower_clustering()
    #Agglomerative()