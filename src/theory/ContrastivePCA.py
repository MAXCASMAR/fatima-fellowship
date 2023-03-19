import numpy as np

class ContrastivePCA:
    def __init__(self, n_components=2, alpha=1.0):
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, A, B):
        A_centered = A - np.mean(A, axis=0)
        B_centered = B - np.mean(B, axis=0)

        C_A = np.cov(A_centered, rowvar=False)
        C_B = np.cov(B_centered, rowvar=False)
        C_cPCA = C_A - self.alpha * C_B

        eigvals, eigvecs = np.linalg.eigh(C_cPCA)
        sorted_indices = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        return X @ self.components_

    def fit_transform(self, A, B):
        self.fit(A, B)
        return self.transform(A)

# Create synthetic data
np.random.seed(42)
A = np.random.normal(0, 1, size=(100, 3))
B = np.random.normal(0, 2, size=(100, 3))

# Apply Contrastive PCA
cPCA = ContrastivePCA(n_components=2, alpha=1.0)
A_cPCA = cPCA.fit_transform(A, B)

# Apply standard PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
A_pca = pca.fit_transform(A)

print("Standard PCA:")
print(A_pca[:5])
print("\nContrastive PCA:")
print(A_cPCA[:5])
