import kmeans_main as kmean
import options_lib as opt

distance = "euclidean"
# distance = "correlation-pca"
# distance = "correlation"
# distance = "dwt"

# target = "series"
target = "components"

#fitting = "eigen_vectors"
fitting = "eigvect_eigval"

weighted_distance = False

observations = 220  # max 220
N = 10  # Number of iterations
normalize_return = True

plot_pca = False
plot_projected_mat = False
plot_clusters = False
plot_fitting = False

file_path=r'P:\Projects\Master Thesis'
workbook_name = "XHRC-datareduced.xlsm"

allocation_table, clustering_cost, exec_time, fitted_coefficients, all_returns, all_projected_clusters = kmean.get_clusters(
                                                                                distance, target, fitting,
                                                                                observations, normalize_return, N,
                                                                                plot_pca, plot_projected_mat,
                                                                                plot_clusters, plot_fitting,
                                                                                file_path, workbook_name,
                                                                                weighted_distance)
print("")
print(allocation_table)
print("")
print(clustering_cost)

opt.options_pricing(fitted_coefficients, all_returns)