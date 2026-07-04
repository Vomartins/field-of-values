import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import ConvexHull


def mv(v):
    return A @ v


def is_normal_matrix(A, epsilon=1e-8, num_tests=3):
    n = A.shape[0]

    print("Running memory-safe normality check...")

    # A.conj().T is the conjugate transpose (required if your matrix has complex numbers)
    A_adj = A.conj().T

    for i in range(num_tests):
        # 1. Generate a random vector
        v = np.random.rand(n)

        # 2. Compute (A* A) v sequentially
        # The @ symbol is the modern Python operator for matrix multiplication
        step1 = A @ v
        A_adj_A_v = A_adj @ step1

        # 3. Compute (A A*) v sequentially
        step2 = A_adj @ v
        A_A_adj_v = A @ step2

        # 4. Find the maximum difference between the two resulting vectors
        max_diff = np.max(np.abs(A_adj_A_v - A_A_adj_v))
        print(f"Test {i + 1}: Maximum error = {max_diff:.2e}")

        if max_diff > epsilon:
            print("Matrix is NOT normal.")
            return False

    print("Matrix IS normal (within epsilon).")
    return True


def Soules():
    return np.array(
        [
            [0.1348, 0.1231, 0.1952, 0.3586, 0.8944],
            [0.2697, 0.2462, 0.3904, 0.7171, -0.4472],
            [0.4045, 0.3693, 0.5855, -0.5976, 0],
            [0.5394, 0.4924, -0.6831, 0, 0],
            [0.6742, -0.7385, 0, 0, 0],
        ]
    )  # Soules Matrix


def num_field(A, theta):
    n = A.shape[0]
    eigvalmax = []
    x_W = []
    y_W = []

    # --- BULLETPROOF CHUNKED MULTIPLICATORS ---
    # These functions process the 14GB disk matrix in 600MB chunks.
    # They guarantee RAM usage stays low.

    def safe_matvec(v_in):
        """Computes A @ v in small row-blocks"""
        res = np.zeros(n, dtype=float)
        chunk = 2000
        for i in range(0, n, chunk):
            res[i : i + chunk] = A[i : i + chunk, :] @ v_in
        return res

    def safe_rmatvec(v_in):
        """Computes A.T @ v in small row-blocks by calculating v @ A"""
        res = np.zeros(n, dtype=float)
        chunk = 2000
        for i in range(0, n, chunk):
            # v @ A is mathematically identical to A.T @ v for 1D vectors,
            # but it prevents NumPy from trying to transpose the 14GB file!
            res += v_in[i : i + chunk] @ A[i : i + chunk, :]
        return res

    print("Computing numerical field boundary points...")
    for k in range(theta.size):
        c1 = np.exp(complex(0, -theta[k]))
        c2 = np.exp(complex(0, theta[k]))

        def Ah_matvec(v):
            # 1. Split the complex vector to prevent the 28GB type-cast crash
            v_real = v.real
            v_imag = v.imag

            # 2. Multiply real and imaginary parts separately (float64 @ float64)
            A_v_real = safe_matvec(v_real)
            A_v_imag = safe_matvec(v_imag)
            A_v = A_v_real + 1j * A_v_imag

            # 3. Same for the transpose conjugate (A is real, so A* = A.T)
            A_adj_v_real = safe_rmatvec(v_real)
            A_adj_v_imag = safe_rmatvec(v_imag)
            A_adj_v = A_adj_v_real + 1j * A_adj_v_imag

            return 0.5 * (c1 * A_v + c2 * A_adj_v)

        # Wrap the operator. dtype=complex is required for the eigensolver
        Ah_op = sp.sparse.linalg.LinearOperator((n, n), matvec=Ah_matvec, dtype=complex)

        # Find the minimum eigenvalue
        l, v_eig = sp.sparse.linalg.eigsh(Ah_op, k=1, which="SA")

        min_eigval = l[0]
        v0 = v_eig[:, 0]

        eigvalmax.append(min_eigval)

        # Memory-safe computation of p = v* A v
        A_v0 = safe_matvec(v0.real) + 1j * safe_matvec(v0.imag)
        p = np.dot(np.conjugate(v0), A_v0)

        x_W.append(p.real)
        y_W.append(p.imag)

        if (k + 1) % 10 == 0:
            print(f"Processed {k + 1}/{theta.size} angles...")

    return np.array(x_W), np.array(y_W), np.array(eigvalmax)


def outer_approx(autoval, theta):
    xq_W = []
    yq_W = []
    for k in range(theta.size - 1):
        delta = theta[k + 1] - theta[k]
        xq_W.append(
            autoval[k] * np.cos(theta[k])
            + (np.sin(theta[k]) / np.sin(delta))
            * (autoval[k] * np.cos(delta) - autoval[k + 1])
        )
        yq_W.append(
            -autoval[k] * np.sin(theta[k])
            + (np.cos(theta[k]) / np.sin(delta))
            * (autoval[k] * np.cos(delta) - autoval[k + 1])
        )
    return xq_W, yq_W


# A = sp.linalg.leslie([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
# A = sp.linalg.toeplitz([1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1 ,1 , 0, 0, 0, 0, 0, 0])
# A = np.random.random((10, 10))
# A = np.array([[1,1,0,0],[0,1,1,1],[1,0,1,0],[1,0,0,1]])
# A = np.array([[1,1,0],[0,1,1],[1,0,1]])

parser = argparse.ArgumentParser(
    description="Calculate Numerical Field of Values from a memory-mapped matrix."
)
parser.add_argument(
    "filename",
    type=str,
    help="Filename of the matrix binary file (e.g., AM_inv_123.dat)",
)
parser.add_argument(
    "--model",
    type=str,
    help="Name of the model that the matrix was extracted from. (Check the matrices directory for available models)",
)
parser.add_argument(
    "--angles", type=int, default=100, help="Number of points for the hull"
)

args = parser.parse_args()

file_path = Path(args.filename)
filename_base = file_path.stem

model = args.model
angles = args.angles

print(f"Model: {model}, Angles: {angles}")

# Slice out the ID by removing the "AM_inv_" prefix
if filename_base.startswith("AM_inv_"):
    matrix_id = filename_base.replace("AM_inv_", "")
else:
    # Fallback just in case you pass a file with a different naming convention
    matrix_id = "unknown_id"

print(f"Matrix ID: {matrix_id}")

print("Loading matrix...")
A = np.memmap(
    f"matrices/{model}/AM_inv_{matrix_id}.dat",
    dtype="float64",
    mode="r",
    shape=(42346, 42346),
)

print("Defining A_operator...")
A_operator = sp.sparse.linalg.LinearOperator((42346, 42346), matvec=mv)

# print(f"Matrix type: {type(A)}")
# print(f"Matrix shape: {A.shape}")
# print(f"Number of non-zero entries: {np.count_nonzero(A)}")
print("Computing eigenvalues...")
eigvalues = sp.sparse.linalg.eigs(
    A_operator, k=100, sigma=0.0, which="LM", return_eigenvectors=False
)
unique, counts = np.unique(eigvalues, return_counts=True)
eig_multiplication = dict(zip(unique, counts))
print(eig_multiplication)
x_eig = [eigvalues[k].real for k in range(len(eigvalues))]
y_eig = [eigvalues[k].imag for k in range(len(eigvalues))]

plt.figure(figsize=(8, 5))
plt.title(f"Eigenvalues - {model}/{matrix_id}", size=18)

plt.grid(True)
plt.scatter(x_eig, y_eig, c="r", marker="o")
plt.tight_layout()

plt.savefig(f"figures/{model}/eigenvalues_{model}_{matrix_id}.png")

# plt.show()

# IsItNormal = is_normal_matrix(A, epsilon=1e-8)

# if IsItNormal:
#     eigval_min = np.min(eigvalues)
#     eigval_max = np.max(eigvalues)
#     points = np.vstack((x_eig, y_eig)).T
#     hull = ConvexHull(points)

#     plt.figure(figsize=(5, 5))
#     plt.title(f"Field of Values - {model}/{matrix_id}", size=18)

#     for simplex in hull.simplices:
#         plt.plot(points[simplex, 0], points[simplex, 1], "k-")

#     plt.scatter(x_eig, y_eig, c="r", marker="o", label="Eigenvalues")
#     plt.legend(loc="best", prop={"size": 12})
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(f"figures/{model}/field_of_values_{model}_{matrix_id}.png")
#     # plt.show()
# else:
#     theta = np.linspace(0, 2 * np.pi, angles)

#     x_W, y_W, eigvalmax = num_field(A, theta)
#     xq_W, yq_W = outer_approx(eigvalmax, theta)

#     points = np.vstack((x_W, y_W)).T
#     outer_points = np.vstack((xq_W, yq_W)).T
#     hull = ConvexHull(points)
#     outer_hull = ConvexHull(outer_points)

#     plt.figure(figsize=(5, 5))
#     plt.title(f"Field of Values - {model}/{matrix_id}", size=18)

#     for simplex in outer_hull.simplices:
#         plt.plot(outer_points[simplex, 0], outer_points[simplex, 1], "b-")

#     for simplex in hull.simplices:
#         plt.plot(points[simplex, 0], points[simplex, 1], "k-")

#     plt.scatter(x_eig, y_eig, c="r", marker="o", label="Eigenvalues")
#     plt.legend(loc="best", prop={"size": 12})
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(f"figures/{model}/field_of_values_{model}_{matrix_id}.png")
