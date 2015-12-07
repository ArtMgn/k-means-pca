import matplotlib.pyplot as plt
import numpy as np

import ClassFile
import distances_lib as dist

__author__ = 'MagnieAr'


# k_means_alloc_table
# Inputs : data projected in PCA space, original dataset, number of dimensions, number of iterations, number of clusters
def k_means_reduced(all_projected_matrix, distance, d, n, k, weighted_distance):

    # Initialization of clusters - according to K parameter
    allocation_table = initial_allocation(k)
    print(allocation_table)
    results = ClassFile.Results()
    all_projected_clusters = create_projected_cluster(all_projected_matrix, allocation_table)

    # Iterations
    for i in range(0, n):

        for c in range(1, k + 1):
            cluster_results = ClassFile.ClusterResults()
            for s in range(1, d + 1):
                if distance == "euclidean":
                    cur_dist = dist.euclidean_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance)
                if distance == "correlation-pca":
                    cur_dist = dist.correlation_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance)
                if distance == "dwt":
                    cur_dist = dist.dwt_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance)

                cluster_results.add_commodity(s)
                cluster_results.add_distance(cur_dist)
            # this object contains all the desired distances
            results.add_cluster_result(cluster_results)

        # Update assignments
        allocation_table = update_allocation(results, allocation_table)  # change prev alloc table based on curr results
        # print(allocation_table)
        results = ClassFile.Results()  # reset results
        all_projected_clusters = create_projected_cluster(all_projected_matrix, allocation_table)

    return allocation_table, all_projected_clusters;


def k_means_pc(all_eigen_vectors, all_projected_matrix, distance, d, n, k, weighted_distance):

    # Initialization of clusters - according to K parameter
    allocation_table = initial_allocation(k)
    print(allocation_table)
    results = ClassFile.Results()
    all_pc_clusters = create_pc_prototype(all_eigen_vectors, all_projected_matrix, allocation_table)

    # Iterations
    for i in range(0, n):

        for c in range(1, k + 1):
            cluster_results = ClassFile.ClusterResults()
            for s in range(1, d + 1):
                if distance == "euclidean":
                    cur_dist = dist.euclidean_distance_pc(all_eigen_vectors, all_projected_matrix,
                                                          all_pc_clusters, s, c, weighted_distance)
                if distance == "correlation-pca":
                    cur_dist = dist.correlation_distance_pc(all_eigen_vectors, all_projected_matrix,
                                                            all_pc_clusters, s, c, weighted_distance)
                if distance == "dwt":
                    # TODO
                    cur_dist = dist.dwt_distance_pc(all_eigen_vectors, all_projected_matrix,
                                                    all_pc_clusters, s, c, weighted_distance)

                cluster_results.add_commodity(s)
                cluster_results.add_distance(cur_dist)
            # this object contains all the desired distances
            results.add_cluster_result(cluster_results)

        # Update assignments
        print "New allocation"
        allocation_table = update_allocation(results, allocation_table)  # change prev alloc table based on curr results
        results = ClassFile.Results()  # reset results
        all_pc_clusters = create_pc_prototype(all_eigen_vectors, all_projected_matrix, allocation_table)

        print "Code book"
        print(allocation_table)

    return allocation_table, all_pc_clusters;


def create_pc_prototype(all_eigen_vectors, all_projected_matrix, allocation_table):

    all_prototypes = []

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    for j in range(0, c):  # clusters
        contribution_1 = []
        contribution_2 = []
        contribution_3 = []
        first_c = []
        second_c = []
        third_c = []
        for p in range(0, r):  # samples within a cluster
            index = allocation_table[p, j]  # get sample number
            if index != 0:
                first_c.append(all_eigen_vectors[index-1][0][0])
                contribution_1.append(all_projected_matrix[index-1].components[0].contribution)

                second_c.append(all_eigen_vectors[index-1][0][1])
                contribution_2.append(all_projected_matrix[index-1].components[1].contribution)

                third_c.append(all_eigen_vectors[index-1][0][2])
                contribution_3.append(all_projected_matrix[index-1].components[2].contribution)

            projected_cluster = ClassFile.ProjectedSeries("def")

        component_1 = ClassFile.Component(np.mean(contribution_1))
        component_1.add_series(np.mean(first_c, axis=0))
        projected_cluster.add_components(component_1)

        component_2 = ClassFile.Component(np.mean(contribution_2))
        component_2.add_series(np.mean(second_c, axis=0))
        projected_cluster.add_components(component_2)

        component_3 = ClassFile.Component(np.mean(contribution_3))
        component_3.add_series(np.mean(third_c, axis=0))
        projected_cluster.add_components(component_3)

        all_prototypes.append(projected_cluster)

    return all_prototypes;


# TODO
def k_means_whole_data(original_samples, d, n, k):

    # Initialization of clusters - according to K parameter
    allocation_table = initial_allocation(k)
    print(allocation_table)
    results = ClassFile.Results()

    # Iterations
    for n in range(0, n):

        for c in range(1, k + 1):
            cluster_results = ClassFile.ClusterResults()
            for s in range(1, d + 1):
                pearson_coeff =0
                dist =0
                cluster_results.add_commodity(s)
                cluster_results.add_distance(dist)
            # this object contains all the desired distances
            results.add_cluster_result(cluster_results)

        # Update assignments
        print "New allocation"
        allocation_table = update_allocation(results, allocation_table)  # change prev alloc table based on curr results
        results = ClassFile.Results()  # reset results
        all_projected_clusters = create_projected_cluster(0, allocation_table)

        print "Code book"
        print(allocation_table)

    return allocation_table, all_projected_clusters;


def initial_allocation(k):

    # items = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0])
    items = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0])
    np.random.shuffle(items)

    allocation_table = np.array([items[0:4], items[4:8], items[8:12], items[12:16]])

    return allocation_table;


def print_distances_sample_to_cluster(results, commodities_number):

    for s in range(0, commodities_number-1):

        d_to_cluster = [(results.cluster_results[c].distances[s], c)
                        for c in range(0, 4)]
        d_to_cluster.sort()
        print(s+1, d_to_cluster)

    return;


def print_distances_cluster_to_sample(results, commodities_number):

    for c in range(0, 4):
        d_to_cluster = [(results.cluster_results[c].distances[s], s+1)
                        for s in range(0, commodities_number)]
        d_to_cluster.sort()
        print(c, d_to_cluster)

    return 0;


def update_allocation(results, allocation_table):

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    clusters = ClassFile.Results()
    for j in range(0, c):  # initialisation
        cluster_results = ClassFile.ClusterResults()
        clusters.add_cluster_result(cluster_results)

    for j in range(0, c):  # clusters
        for p in range(0, r):  # samples within a cluster
            index = allocation_table[p, j]  # get sample number
            if index != 0:
                # Get all distances corresponding to index, associated with cluster number (0 to k-1)
                d_to_cluster = [(results.cluster_results[i].distances[index-1], i)
                                for i in range(0, c)]
                d_to_cluster.sort()  # Sort the distance in ascending order
                t = 0
                # This line ensures that if a cluster is full (number of sample inside = c), we fill the next one
                # based on d_to_cluster (which has just been sorted)
                while clusters.cluster_results[d_to_cluster[t][1]].number >= c:
                    t += 1
                # Store the new results in a cluster with available space
                clusters.cluster_results[d_to_cluster[t][1]].add_commodity(index)

    new_alloc_table = np.zeros((r, c), dtype=np.int)

    for j in range(0, c):
        for p in range(0, clusters.cluster_results[j].number):
            new_alloc_table[p][j] = int(clusters.cluster_results[j].commodities[p])

    return new_alloc_table;


def create_projected_cluster(all_projected_matrix, allocation_table):

    all_projected_clusters = []

    # TODO : refactor the code

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    for j in range(0, c):  # clusters
        contribution_1 = []
        contribution_2 = []
        contribution_3 = []
        first_c = []
        second_c = []
        third_c = []
        for p in range(0, r):  # samples within a cluster
            index = allocation_table[p, j]  # get sample number
            if index != 0:
                first_c.append(all_projected_matrix[index-1].components[0].series[0])
                contribution_1.append(all_projected_matrix[index-1].components[0].contribution)

                second_c.append(all_projected_matrix[index-1].components[1].series[0])
                contribution_2.append(all_projected_matrix[index-1].components[1].contribution)

                third_c.append(all_projected_matrix[index-1].components[2].series[0])
                contribution_3.append(all_projected_matrix[index-1].components[2].contribution)

        projected_cluster = ClassFile.ProjectedSeries("def")

        component_1 = ClassFile.Component(np.mean(contribution_1))
        component_1.add_series(np.mean(first_c, axis=0))
        projected_cluster.add_components(component_1)

        component_2 = ClassFile.Component(np.mean(contribution_2))
        component_2.add_series(np.mean(second_c, axis=0))
        projected_cluster.add_components(component_2)

        component_3 = ClassFile.Component(np.mean(contribution_3))
        component_3.add_series(np.mean(third_c, axis=0))
        projected_cluster.add_components(component_3)

        all_projected_clusters.append(projected_cluster)

    return all_projected_clusters;


def cost_of_clustering(all_projected_clusters, all_projected_matrix, allocation_table, weighted_distance):

    cost = np.zeros(4)
    for i in range(0, 4):
        for j in range(0, 4):
            if allocation_table[j][i] != 0:
                cost[i] += dist.dwt_distance(all_projected_matrix, all_projected_clusters,
                                               allocation_table[j][i]-1, i, weighted_distance)

    return cost;


def display_final_clusters(all_projected_clusters, all_projected_matrix, allocation_table):

    plt.figure(0)

    colors = ["r", "m", "g", "y", "b"]

    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        plt.plot(all_projected_clusters[i].components[0].series[0], label='centroid')
        plt.title('Cluster {0}'.format(i+1))
        for j in range(0, 4):
            if allocation_table[j][i] != 0:
                plt.plot(all_projected_matrix[allocation_table[j][i] - 1].components[0].series[0],
                         '{0}--'.format(colors[j]),
                         label='cmdty {0}'.format(allocation_table[j][i]))
                plt.legend(loc='upper right')

    plt.show()

    return;


def display_final_clusters_pc(all_eigen_vectors, all_pc_clusters, allocation_table, components_number):

    plt.figure(0)

    colors = ["r", "m", "g", "y", "b"]

    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        plt.plot(all_pc_clusters[i].components[0].series[0], 'b', label='centroid, PC1')
        for k in range(1, components_number):
            plt.plot(all_pc_clusters[i].components[k].series[0], '{0}'.format(colors[k]), label='centroid, PC{0}'.format(k+1))
        plt.title('Cluster {0}'.format(i+1))
        for j in range(0, 4):
            if allocation_table[j][i] != 0:
                plt.plot(all_eigen_vectors[allocation_table[j][i] - 1][0][0],
                         '{0}--'.format(colors[j]),
                         label='cmdty {0}'.format(allocation_table[j][i]))
                for k in range(1, components_number):
                    plt.plot(all_eigen_vectors[allocation_table[j][i] - 1][0][k],
                             '{0}--'.format(colors[j]))
                plt.legend(loc='upper right')

    plt.show()

    return;