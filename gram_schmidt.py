import numpy as np

def GramSchmidt(A):
    B = []
    (N, M) = A.shape
    for i in range(M):
        a = A[:,i]
        m = len(B)
        for j in range(m):
            b = np.dot(B[j],a)*B[j]
            a = a-b
        a = a/np.sqrt(np.dot(a,a))
        B.append(a)
    return B

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def base_canon_comp(x,y,M):
    B = np.concatenate((x.T,y.T),axis=1)
    for i in range(M-2):
        aux = np.zeros((M,1))
        aux[i+2] = 1
        B = np.concatenate((B,aux),axis=1)


    return B

x = np.array([[1, 1, 1, 1, 1]])
y = np.array([[2, 0, 2, 0, 2]])

B = base_canon_comp(x,y,x.size)

print(B)
print(gram_schmidt_columns(B))
print(np.dot(B,x.T))