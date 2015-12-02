import os
import time

import numpy as np
import sklearn.preprocessing as pp
from xlrd import open_workbook

import ClassFile
import distances_lib as dist
import function_fitting_lib as ff
import kmeans_lib as km
import pca_lib as pc

__author__ = 'MagnieAr'


file_path=r'P:\Projects\Master Thesis'

Book = open_workbook(os.path.join(file_path,"XHRC-datareduced.xlsm"), on_demand=True)

means_and_scatters = np.zeros((4, Book.nsheets))
time_scatter = np.zeros((Book.nsheets, 10))

samples = ClassFile.Samples()

big_matrix = np.zeros((220, 0))

# Control parameters

pca_mode = True
plot_pca = True
plot_projected_mat = True
fit_mode = True
plot_clusters = True

start = time.time()
all_projected_matrix = []
"""
    Computation of initial parameters, means and scatters of the data sample
"""

for sht_idx in range(0, Book.nsheets):
    sheet_name = Book.sheet_by_index(sht_idx).name
    sheet = Book.sheet_by_name(sheet_name)

    # num_rows = sheet.nrows - 1
    num_rows = 120
    # num_rows = 220
    num_cols = sheet.ncols - 1

    all_maturities = [0 for x in range(num_rows)]

    scatter_matrix = np.zeros((num_cols, num_rows))
    scatter_returns = np.zeros((num_cols, num_rows-1))

    for col_idx in range(1, num_cols+1):
            for row_idx in range(1, num_rows+1):
                    cell_obj = sheet.cell(row_idx, col_idx)
                    all_maturities[row_idx-1] = cell_obj.value
            scatter_matrix[col_idx-1] = all_maturities

    X = pp.normalize(scatter_matrix, norm='l2', axis=0, copy=True)  # First, we need to gather all the data in one matrix in order to whiten the prices
                                                                    #(improve k-means performance and remove correlation biased)

    # scatter_matrix = X

    for col_idx in range(0, num_cols):
        for row_idx in range(0, num_rows):
                val = scatter_matrix.T[row_idx][col_idx]
                all_maturities[row_idx] = val
                if row_idx > 0:
                    scatter_returns[col_idx][row_idx-1] = (all_maturities[row_idx] - all_maturities[row_idx-1]) / all_maturities[row_idx-1]

    R = scatter_returns.T
    # whiten(R)

    """
        Principal component decomposition before k-means
    """

    if pca_mode:
        eig_vectors, ProjectedSample, eig_values = pc.pca(R, sheet_name, sht_idx, num_rows, plot_pca)
        if fit_mode:
            results = ff.function_fitting(1, eig_vectors, eig_values, sht_idx, sheet_name)

        projected_matrix = eig_vectors.dot(R.T)

        for j in range(0,3):
                ProjectedSample.components[j].add_series(projected_matrix[j, :])

        all_projected_matrix.append(ProjectedSample)

        if plot_projected_mat:
            pc.plot_matrix(sht_idx, sheet_name, projected_matrix, R)

    # plt.show()
    # pylab.savefig("PCA space " + sheet_name + ".png", bbox_inches='tight')

    mu_siT, omega, M = dist.data_mu(R)

    scat, sample_temp = dist.scatter(R, omega, sheet_name)

    sample = ClassFile.Sample(mu_siT, scat, omega, M, sheet_name)
    sample.add_all_scatter_maturity(sample_temp.scatter_maturity)
    samples.add_sample(sample)

"""
    Computation of clusters' initial mean and scatter
"""

#  clusters_mu = clust_mu(samples, allocation_table)
#  clusters = clust_scatter(samples, clusters_mu, allocation_table)


"""
    Computation of the distance and update of the allocation table - main algorithm
"""

N = 20  # Number of iterations
K = 4

#  Results = ClassFile.Results()
allocation_table, all_projected_clusters = km.k_means_alloc_table(all_projected_matrix, samples, Book.nsheets, N, K)
#features = all_projected_matrix
#whitened = whiten(features)
#book = np.array((whitened[0],whitened[2]))
#kmeans(whitened,book)
#test = kmeans()
print ""
print "Code book"
print(allocation_table)

costs = km.cost_of_clustering(all_projected_clusters, all_projected_matrix, allocation_table)

print(costs, sum(costs))

if plot_clusters:
    km.display_final_clusters(all_projected_clusters, all_projected_matrix, allocation_table)

"""

    PART II - Volatility fitting

"""

end = time.time()

print end - start
