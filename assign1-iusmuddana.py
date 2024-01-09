import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

datafile = sys.argv[1]
data = pd.read_csv(datafile,  names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"])

features = data.iloc[:, :10]
labels = data.iloc[:, 10]

# Apply z-normalization to the features
df = (features - features.mean()) / features.std()

mean_values = df.mean()
centered_data = df - mean_values

# Compute the sample covariance matrix using the outer product
n = len(df)
covariance_matrix_manual = (1 / n) * centered_data.T.dot(centered_data)
covariance_matrix_numpy = np.cov(df, rowvar=False, bias=True)
print('#'*100)
print("Covariance Matrix (Computed Manually):\n", covariance_matrix_manual)
print("\nCovariance Matrix (Using numpy.cov):\n", covariance_matrix_numpy)
print("\n")

# Power iteration to compute the dominant eigenvector and eigenvalue
x0 = np.random.rand(df.shape[1])
max_iterations = 1000
threshold = 0.000001
norm_difference = float('inf')

for i in range(max_iterations):
    xi = np.dot(covariance_matrix_manual, x0)
    mi = np.argmax(np.abs(xi))
    xi_scaled = xi / xi[mi]
    norm_difference = np.linalg.norm(xi_scaled - x0)
    if norm_difference < threshold:
        break
    x0 = xi_scaled

# Normalize the final eigenvector to have unit length
eigenvector = xi_scaled / np.linalg.norm(xi_scaled)

# Compute the dominant eigenvalue using the ratio of xi,mi / xi−1,mi−1
dominant_eigenvalue = xi[mi] / x0[mi]

# Verify the result using numpy linalg.eig function
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix_manual)
dominant_eigenvalue_numpy = np.max(eigenvalues)
dominant_eigenvector_numpy = eigenvectors[:, np.argmax(eigenvalues)]
print('#'*100)
print("Dominant Eigenvalue (Power Iteration Method):", dominant_eigenvalue)
print("Dominant Eigenvector (Power Iteration Method):\n", eigenvector)
print("\nDominant Eigenvalue (numpy.linalg.eig):", dominant_eigenvalue_numpy)
print("Dominant Eigenvector (numpy.linalg.eig):\n", dominant_eigenvector_numpy)
print("\n")

sortedInd = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sortedInd[:2]]
projected_data = np.dot(df, top_eigenvectors)

variance_in_projected_subspace = np.var(projected_data, axis=0)
print('#'*100)
print("Variance of Data Points in Projected Subspace:")
print(variance_in_projected_subspace)
print("\n")

eigenvalues_diag = np.diag(eigenvalues)

covariance_decomposition = np.dot(eigenvectors, np.dot(eigenvalues_diag, eigenvectors.T))
print('#'*100)
print("Covariance Matrix in Eigen-decomposition Form:")
print(covariance_decomposition)
print("\n")

def compute_mse(data_points, eigenvectors):
    # Project the data points onto the subspace spanned by the first two eigenvectors
    proj_data = np.dot(data_points, eigenvectors)
    recons_data = np.dot(proj_data, eigenvectors.T)
    mse = np.mean(np.sum((df - recons_data)**2, axis=1))
    return mse

sum_of_eigenvalues_except_first_two = np.sum(np.sort(np.linalg.eigvals(covariance_matrix_manual))[:-2])
mse_projection = compute_mse(df, top_eigenvectors)
print('#'*100)
print("Sum of Eigenvalues (except the first two):", sum_of_eigenvalues_except_first_two)
print("MSE for the Projection:", mse_projection)
print("MSE equals Sum of Eigenvalues (except the first two):", np.isclose(mse_projection, sum_of_eigenvalues_except_first_two))
print("\n")
# Project the data points onto the subspace spanned by the first two eigenvectors
projected_data = np.dot(df, top_eigenvectors)

# Plot the data points using different colors for each class
unique_classes = labels.unique()
colors = plt.cm.get_cmap('tab10', len(unique_classes))
plt.figure(figsize=(10, 6))
for i, class_label in enumerate(unique_classes):
    class_indices = labels[labels == class_label].index
    plt.scatter(projected_data[class_indices, 0], projected_data[class_indices, 1], label=f'Class {class_label}', color=colors(i))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Data Points Projected onto First Two Principal Components')
plt.show()

def pca_algorithm(data, variance_threshold=0.95):
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    num_principal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    principal_vectors = eigenvectors[:, :num_principal_components]
    return principal_vectors

principal_vectors = pca_algorithm(df, variance_threshold=0.95)
print('#'*100)
# Print the coordinates of the first 10 data points using the new basis vectors
print("Coordinates of the first 10 data points in the new basis:")
projected_data_10 = np.dot(df.iloc[:10], principal_vectors)
print(projected_data_10)