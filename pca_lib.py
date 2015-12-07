import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import ClassFile


def pca(returns_matrix, sheet_name, sht_idx, num_rows, plot_pca, plot_projected_mat):

    pca_out = PCA(n_components=10)
    pca_out.fit(returns_matrix)
    pca_out.transform(returns_matrix)
    PCA(copy=True, n_components=10, whiten=False)
    eig_values = pca_out.explained_variance_
    eig_vectors = pca_out.components_

    diag_mat = np.diag(np.sqrt(eig_values))
    vol_matrix = eig_vectors.dot(diag_mat)

    projected_sample = ClassFile.ProjectedSeries(sheet_name)

    sum_pair = 0
    for i in eig_values:
        sum_pair = sum_pair + i

    contribution = np.zeros(len(eig_values))
    z = 0

    for i in eig_values:
        contribution[z] = i / sum_pair
        z += 1

    for j in range(0, 3):
        component = ClassFile.Component(contribution[j])
        projected_sample.add_components(component)

    if plot_pca:
        plot_pc(eig_vectors, sheet_name, sht_idx, num_rows, True)

    # Project the returns in the PCA space
    projected_matrix = eig_vectors.dot(returns_matrix.T)

    # Plot the projected matrix on request
    if plot_projected_mat:
        plot_matrix(sht_idx+1, sheet_name, projected_matrix, returns_matrix)

    # Return only the first three Eigen-vectors
    return eig_vectors[0:3], projected_sample, eig_values, projected_matrix;


def plot_pc(eig_vectors, sheet_name, sht_idx, num_rows, default):

    plt.figure(sht_idx)
    if default:
        plt.plot(eig_vectors[0:3].T)
    else:
        plt.plot(eig_vectors.T)
    min = np.min(eig_vectors[0:3, 0:10])
    max = np.max(eig_vectors[0:3, 0:10])
    plt.plot(eig_vectors[0, 0:10], 'o', markersize=7, color='blue', alpha=0.5, label='PCA 1')
    plt.plot(eig_vectors[1, 0:10], '^', markersize=7, color='green', alpha=0.5, label='PCA 2')
    plt.plot(eig_vectors[2, 0:10], 'x', markersize=7, color='red', alpha=0.5, label='PCA 3')
    plt.xlim([1, 10])
    plt.ylim([min - 0.05, max + 0.05])
    plt.xlabel('Time-to-maturity')
    plt.ylabel('Eigen Vectors - Loading factors')
    plt.legend()
    plt.title('PCA ' + sheet_name + ' - ' + str(num_rows) + ' months look-back')
    plt.show()

    return;


def plot_matrix(sht_idx, sheet_name, projected_matrix, R):

    plt.figure(sht_idx)
    plt.subplot(211)
    plt.plot(projected_matrix.T)
    plt.ylim([-1,1])
    plt.ylabel('Returns')
    plt.legend()
    plt.title('Return Matrix in PCA space ' + sheet_name)

    plt.subplot(212)
    plt.plot(R)
    plt.ylim([-1,1])
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('Return Matrix ' + sheet_name)
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)

    return;