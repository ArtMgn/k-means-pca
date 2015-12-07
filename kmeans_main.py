import os
import time

import numpy as np
from xlrd import open_workbook

import ClassFile
import distances_lib as dist
import function_fitting_lib as ff
import kmeans_lib as km
import pca_lib as pc
import data_getter as dg

__author__ = 'MagnieAr'


def get_clusters(distance, target, fitting, observations, normalize_return, n, plot_pca,
                 plot_projected_mat, plot_clusters, plot_fitting, file_path, workbook_name, weighted_distance):

    book = open_workbook(os.path.join(file_path, workbook_name), on_demand=True)
    sheets_nbr = book.nsheets

    start = time.time()

    if distance == "correlation":
        samples = ClassFile.Samples()
    else:
        all_projected_matrix = []
        all_eigen_vectors = []

    all_coefficients = []
    all_returns = []

    """
        Computation of initial parameters, means and scatters of the data sample
    """

    for sht_idx in range(0, sheets_nbr):

        # Get the matrix of returns
        R, sheet_name = dg.get_returns(sht_idx, normalize_return, observations)
        all_returns.append(R)

        if distance == "correlation":
            """
                Get the parameters used in correlation-based distance
            """
            mu_siT, omega, M = dist.data_mu(R)

            scat, sample_temp = dist.scatter(R, omega, sheet_name)

            sample = ClassFile.Sample(mu_siT, scat, omega, M, sheet_name)
            sample.add_all_scatter_maturity(sample_temp.scatter_maturity)
            samples.add_sample(sample)

        """
            Principal component decomposition before k-means
        """

        # Get the projected samples and the eigen vectors
        eig_vectors, projected_sample, eig_values, projected_matrix = pc.pca(R, sheet_name, sht_idx,
                                                                             observations, plot_pca, plot_projected_mat)

        all_eigen_vectors.append((eig_vectors, eig_values))

        # Store the first 3 projected component in  projected_sample
        for j in range(0,3):
                projected_sample.components[j].add_series(projected_matrix[j, :])

        # Store all the series projected in the PCA space into a bigger object
        all_projected_matrix.append(projected_sample)

        # plt.show()
        # pylab.savefig("PCA space " + sheet_name + ".png", bbox_inches='tight')

        """
            Fitting of principal components
        """

        # Get the coefficients from the fitting of the curves
        coefficients, v = ff.function_fitting(fitting, 1, eig_vectors, eig_values, sht_idx, sheet_name, plot_fitting)
        all_coefficients.append(coefficients)

    """
        Computation of clusters' initial mean and scatter
    """

    #  clusters_mu = clust_mu(samples, allocation_table)
    #  clusters = clust_scatter(samples, clusters_mu, allocation_table)

    """
        Computation of the distance and update of the allocation table - main algorithm
    """


    K = 4
    component_number = 2

    if distance == "correlation":
        allocation_table = km.k_means_whole_data(samples, sheets_nbr, n, K)  # TODO
        costs = 0
    else:
        if target == "series":
            allocation_table, all_projected_clusters = km.k_means_reduced(all_projected_matrix, distance,
                                                                          sheets_nbr, n, K, weighted_distance)
            costs = km.cost_of_clustering(all_projected_clusters, all_projected_matrix, allocation_table, weighted_distance)
        elif target == "components":
            allocation_table, all_pc_clusters = km.k_means_pc(all_eigen_vectors, all_projected_matrix, distance,
                                                              sheets_nbr, n, K, weighted_distance)
            costs = 0

        if plot_clusters:
            if target == "series":
                km.display_final_clusters(all_projected_clusters, all_projected_matrix, allocation_table)
            elif target == "components":
                km.display_final_clusters_pc(all_eigen_vectors, all_pc_clusters,
                                             allocation_table, component_number)

    end = time.time()

    if distance == "correlation":
        ret = allocation_table, costs, end - start, all_coefficients, all_returns
    else:
        if target == "series":
            ret = allocation_table, costs, end - start, all_coefficients, all_returns, all_projected_clusters
        elif target == "components":
            ret = allocation_table, costs, end - start, all_coefficients, all_returns, all_pc_clusters

    return ret;