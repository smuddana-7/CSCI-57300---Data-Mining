# Import Libraries
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import multivariate_normal

# Data Loading Function
def load_data(filename):
    """Load the dataset from the provided filename and separate features from labels."""
    df = pd.read_csv(filename, names=['sepal_length', 'sepel_width', 'petal_length', 'petal_width', 'label'], index_col=False)
    label = df.label.values
    df.drop('label', axis=1, inplace=True)
    return df.values, label


# Initialization Function for EM
def init_em_params(X, k):
    """Initialize the parameters (means, covariances, and weights) for the EM algorithm."""
    n = len(X)
    indices = np.arange(n)
    means = [np.mean(X[indices[i::k]], axis=0) for i in range(k)]
    covariances = [np.cov(X[indices[i::k]].T) for i in range(k)]
    weights = np.ones(k) / k
    return means, covariances, weights


# EM Algorithm Function
def gmm_em(X, k, initial_means, initial_covariances, initial_weights, tolerance=1e-6, max_iterations=1000):
    """Implement the EM algorithm for Gaussian Mixture Models."""
    iters = 0
    means = initial_means
    covariances = initial_covariances
    weights = initial_weights
    
    for iteration in range(max_iterations):
        # Expectation-step: Compute responsibilities
        responsibilities = np.zeros((len(X), k))
        for i in range(k):
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        
        # Maximization-step: Update cluster parameters
        Nk = responsibilities.sum(axis=0)
        weights = Nk / len(X)
        means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        for i in range(k):
            diff = X - means[i]
            covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]

        # Convergence check
        if np.linalg.norm(means - initial_means) < tolerance: 
            iters = iteration
            break        
        initial_means = means
        
    return means, covariances, weights, responsibilities, iters

# Results Printing Function
def display_results(means, covariances, responsibilities, iterations, X, k):
    """Display the clustering results including means, covariances, memberships, and cluster sizes."""
    # Display means
    norms = np.linalg.norm(means, axis=1)
    mean_df = pd.DataFrame(means, columns=['Dim1', 'Dim2', 'Dim3', 'Dim4'])
    mean_df['Norm'] = norms
    sorted_means = mean_df.sort_values(by='Norm').drop(columns='Norm')
    print("\nMean:")
    for row in range(len(sorted_means)):
        print(list(sorted_means.iloc[row, :]))
    sorted_indices = list(sorted_means.index)

    # Display covariance matrices
    print("\nCovariance Matrices:")
    for i in sorted_indices:
        print(covariances[i])
        print("\n")
    
    print("Iteration count= ", iterations)

    # Display cluster memberships
    cluster_assign_sorted = np.argmax(responsibilities[:, sorted_indices], axis=1)
    cluster_mem = {i: [] for i in range(k)}
    for i, cluster_idx in enumerate(cluster_assign_sorted):
        cluster_mem[cluster_idx].append(i)
    cluster_mem_sorted = {i: sorted(cluster_mem[i]) for i in cluster_mem}
    print("\nCluster Membership:")
    for key in cluster_mem_sorted.keys():
        for x in cluster_mem_sorted[key]:
            print(X[x], end=",")
        print()

    # Display cluster sizes
    cluster_sizes = [len(members) for members in cluster_mem_sorted.values()]
    print("\nSize:", " ".join(map(str, cluster_sizes)))

# Purity Calculation Function
def compute_purity(cluster_assignments, labels):
    """Compute the purity score of the clustering."""
    cluster_label_map = pd.crosstab(index=cluster_assignments, columns=labels).idxmax(axis=1)
    cluster_assignments_series = pd.Series(cluster_assignments)
    mapped_labels = cluster_assignments_series.map(cluster_label_map)
    purity_score = (mapped_labels == labels).mean()
    return purity_score


if len(sys.argv) < 2:
    print("No filename provided.")
    sys.exit(1)

filename = sys.argv[1]
X, label = load_data(filename)
k = 3

initial_means, initial_covariances, initial_weights = init_em_params(X, k)
final_means, final_covariances, weights, responsibilities, iters = gmm_em(X, k, initial_means, initial_covariances, initial_weights)

display_results(final_means, final_covariances, responsibilities, iters, X, k)

c_df = np.ascontiguousarray(X)
c_f_means = np.ascontiguousarray(final_means)
cluster_assign = pairwise_distances_argmin_min(c_df, c_f_means)[0]

purity_score = compute_purity(cluster_assign, label)
print(f"\nPurity: {purity_score}")