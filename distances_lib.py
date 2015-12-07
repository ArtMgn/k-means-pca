import numpy as np
import scipy.spatial as scp
import scipy.stats as st
from scipy.cluster.vq import whiten

import ClassFile

__author__ = 'magniear'


def data_mu(scatter_returns):
    # Definitive

    T = len(scatter_returns[0])  # Columns = maturities
    M = len(scatter_returns)  # Rows = samples

    one_T = np.ones((T, 1))
    one_M = np.ones((M, 1))

    omega = np.dot(scatter_returns, one_T)
    omega = np.dot(one_M.T, omega)
    omega /= M

    mu = float(omega / T)

    #  print ('mu ', mu)

    return float(mu), float(omega), M;


def scatter(scatter_returns, omega, name):

    # Definitive
    t = len(scatter_returns[0])  # Columns = maturities
    m = len(scatter_returns)  # Rows = samples

    unit = np.zeros((t, 1))
    one_vec = np.ones((m, 1))

    b = ((1 / float(t)) * (float(omega) / float(m))**2)

    a = 0

    sample_temp = ClassFile.Sample(0, 0, 0, 0, "")

    for i in range(0, t):
        scatter_maturity = ClassFile.ScatterMaturity(0, 0)
        unit[i][0] = 1  # Create unit vector

        s_t = np.dot(scatter_returns, unit)
        s_t = np.dot(one_vec.T, s_t)

        scatter_maturity.scatter = float(s_t)
        scatter_maturity.maturity = i
        sample_temp.add_scatter_maturity(scatter_maturity)

        a += float(s_t)**2
        unit = np.zeros((t, 1))

    scat = float(a) - float(b)

    scat = float(np.sqrt(scat))

    #  print('scat ' + name, scat)

    return scat, sample_temp;


def clust_mu(samples, allocation_table):

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    sum_mu = 0
    sum_m = 0

    clusters = ClassFile.Clusters()

    for i in range(0,  c):
        cluster_o = ClassFile.Cluster(0, 0)
        for j in range(0, r):
            index = allocation_table[j, i]
            if index != 0:
                sum_mu += samples.samples[index-1].size * samples.samples[index-1].omega
                sum_m += samples.samples[index-1].size
        cluster_o.mean = (sum_mu / float(sum_m)) / 10  # sum(m*omega)/sum(m)/T
        cluster_o.scatter = 0
        # print('clust mean : ' + str(cluster_o.mean))
        clusters.add_cluster(cluster_o)
        sum_mu = 0
        sum_m = 0

    return clusters;


def clust_scatter(samples, clusters, allocation_table, n):

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    time_scat_square = 0
    mat_scatter = 0

    for j in range(0, c):  # clusters
        for t in range(0, 10):  # maturities
            for p in range(0, r):  # samples within a cluster
                index = allocation_table[p, j]
                if index != 0:
                    time_scat_square += samples.samples[index-1].scatter_maturity[t].scatter
            mat_scatter += time_scat_square**2
            time_scat_square = 0
        clusters.clusters[j].scatter = np.sqrt(mat_scatter - 10 * clusters.clusters[j].mean**2)
        mat_scatter = 0
        if n == 0 or n == 4999:
            print('clust scatter : ' + str(clusters.clusters[j].scatter))

    # Normalize clusters' scatter
    vec = np.zeros(4)
    for j in range(0, c):
        vec[j] = clusters.clusters[j].scatter

    whiten(vec)
    for j in range(0, c):
        clusters.clusters[j].scatter = vec[j]

    return clusters;


def pearson_coeff(allocation_table, samples, clusters, sample_index, cluster_index):

    c = len(allocation_table[0])  # Columns
    r = len(allocation_table)  # Rows

    time_scat = 0
    sum_size = 0
    sum_num = 0

    for t in range(0, 10):  # maturities
        for p in range(0, r):  # samples within a cluster
            index = allocation_table[p, cluster_index - 1]   # get sample reference within the cluster
            if index != 0:
                time_scat += samples.samples[index-1].scatter_maturity[t].scatter
                sum_size += samples.samples[index-1].size

        sample_term = samples.samples[sample_index-1].scatter_maturity[t].scatter - samples.samples[sample_index-1].size * samples.samples[sample_index-1].mean
        cluster_term = time_scat - sum_size * clusters.clusters[cluster_index - 1].mean
        sum_num += sample_term * cluster_term
        #  print('sample/cluster term', sample_term, cluster_term, sum_num)
        time_scat = 0
        sum_size = 0

    denom = samples.samples[sample_index-1].scatter * clusters.clusters[cluster_index - 1].scatter
    coefficient = sum_num / np.sqrt(denom)
    #  print('index sample,ind clu, samp scatt, clust scat, coeff, dist', sample_index, cluster_index, samples.samples[sample_index-1].scatter, clusters.clusters[cluster_index - 1].scatter, coefficient, d1(coefficient))

    return coefficient;


def euclidean_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2

        eucl_dist = scp.distance.euclidean(all_projected_matrix[s-1].components[i].series[0],
                                           all_projected_clusters[c-1].components[i].series[0])

        if weighted_distance:
            dist += eucl_dist * weight
        else:
            dist += eucl_dist

    return dist;


def euclidean_distance_pc(all_eigen_vectors, all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2

        eucl_dist = scp.distance.euclidean(all_eigen_vectors[s-1][0][i],
                                           all_projected_clusters[c-1].components[i].series[0])

        if weighted_distance:
            dist += eucl_dist * weight
        else:
            dist += eucl_dist

    return dist;


def correlation_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2

        # Get correlation coeff
        pears_coeff = st.pearsonr(all_projected_matrix[s-1].components[i].series[0],
                                           all_projected_clusters[c-1].components[i].series[0])
        # Get the distance
        corr_dist = d1(pears_coeff[0])

        if weighted_distance:
            dist += corr_dist * weight
        else:
             dist += corr_dist

    return dist;


def correlation_distance_pc(all_eigen_vectors, all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2

        pears_coeff = st.pearsonr(all_eigen_vectors[s-1][0][i],
                                           all_projected_clusters[c-1].components[i].series[0])
        corr_dist = d1(pears_coeff[0])

        if weighted_distance:
            dist += corr_dist * weight
        else:
             dist += corr_dist

    return dist;


def d1(p_coeff):

    return 2 * (1 - abs(p_coeff));


def dwt_distance(all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    weight = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2
        dwt_dist = dtw_dist(all_projected_matrix[s-1].components[i].series[0],
                            all_projected_clusters[c-1].components[i].series[0], 24)

        if weighted_distance:
            dist += dwt_dist * weight
        else:
            dist += dwt_dist

    return dist;


def dwt_distance_pc(all_eigen_vectors, all_projected_matrix, all_projected_clusters, s, c, weighted_distance):

    dist = 0
    weight = 0
    for i in range(0, 3):
        weight = (all_projected_matrix[s-1].components[i].contribution +
                  all_projected_clusters[c-1].components[i].contribution)/2

        dwt_dist = dtw_dist(all_eigen_vectors[s-1][0][i],
                            all_projected_clusters[c-1].components[i].series[0], 24)

        if weighted_distance:
            dist += dwt_dist * weight
        else:
             dist += dwt_dist

    return dist;


def dtw_dist(s1, s2, w):

    dtw = {}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            dtw[(i, j)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = (s1[i]-s2[j])**2
            dtw[(i, j)] = dist + min(dtw[(i-1, j)], dtw[(i, j-1)], dtw[(i-1, j-1)])

    return np.sqrt(dtw[len(s1)-1, len(s2)-1])