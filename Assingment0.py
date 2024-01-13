import numpy as np
import time
import threading

def mul(res, A, B):
    n = len(A)
    if n <= 128:  # Most optimal
            res[:] = np.dot(A, B)
    else:
        mid = n // 2
        P = np.zeros((7*mid,mid))

        mul(P[:mid,:],A[:mid, :mid] + A[mid:, mid:], B[:mid, :mid] + B[mid:, mid:])
        mul(P[mid:2*mid,:],A[mid:, :mid] + A[mid:, mid:], B[:mid, :mid])
        mul(P[2*mid:3*mid,:],A[:mid, :mid], B[:mid, mid:] - B[mid:, mid:])
        mul(P[3*mid:4*mid,:],A[mid:, mid:],  B[mid:, :mid] - B[:mid, :mid])
        mul(P[4*mid:5*mid,:],A[:mid, :mid] + A[:mid, mid:], B[mid:, mid:])
        mul(P[5*mid:6*mid,:],A[mid:, :mid] - A[:mid, :mid], B[:mid, :mid] + B[:mid, mid:])
        mul(P[6*mid:7*mid,:],A[:mid, mid:] - A[mid:, mid:],  B[mid:, :mid] + B[mid:, mid:])

        res[:] = np.vstack((np.hstack((P[:mid,:] + P[3*mid:4*mid,:] - P[4*mid:5*mid,:] + P[6*mid:7*mid,:], P[2*mid:3*mid,:] + P[4*mid:5*mid,:])), np.hstack((P[mid:2*mid,:] + P[3*mid:4*mid,:], P[:mid,:] - P[mid:2*mid,:] + P[2*mid:3*mid,:] + P[5*mid:6*mid,:]))))

def calc(B,C,D):
    n = B.shape[0]
    if (n <= 2):
        return np.dot(B,C+D)
    mid = n//2
    P = np.zeros((7*mid,mid))

    threads = [threading.Thread(target=mul, args=(P[:mid,:], B[:mid,:mid] + B[mid:,mid:],C[:mid, :mid]+D[:mid, :mid]+C[mid:, mid:]+D[mid:, mid:])),
               threading.Thread(target=mul, args=(P[mid:2*mid,:], B[mid:,:mid]+B[mid:,mid:],C[:mid, :mid]+D[:mid, :mid])),
               threading.Thread(target=mul, args=(P[2*mid:3*mid,:], B[:mid,:mid],(C[:mid, mid:]+D[:mid, mid:])-(C[mid:, mid:]+D[mid:, mid:]))),
               threading.Thread(target=mul, args=(P[3*mid:4*mid,:], B[mid:,mid:],(C[mid:, :mid]+D[mid:, :mid])-(C[:mid, :mid]+D[:mid, :mid]))),
               threading.Thread(target=mul, args=(P[4*mid:5*mid,:], B[:mid,:mid]+B[:mid,mid:],C[mid:, mid:]+D[mid:, mid:])),
               threading.Thread(target=mul, args=(P[5*mid:6*mid,:], B[mid:,:mid]-B[:mid,:mid],C[:mid, :mid]+D[:mid, :mid]+C[:mid, mid:]+D[:mid, mid:])),
               threading.Thread(target=mul, args=(P[6*mid:7*mid,:],B[:mid,mid:]-B[mid:,mid:],C[mid:, :mid]+D[mid:, :mid]+C[mid:, mid:]+D[mid:, mid:]))
               ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return np.vstack((np.hstack((P[:mid,:] + P[3*mid:4*mid,:] - P[4*mid:5*mid,:] + P[6*mid:7*mid,:], P[2*mid:3*mid,:] + P[4*mid:5*mid,:])), np.hstack((P[mid:2*mid,:] + P[3*mid:4*mid,:], P[:mid,:] - P[mid:2*mid,:] + P[2*mid:3*mid,:] + P[5*mid:6*mid,:]))))


N = 2048   
B = np.random.randint(low=1,high=5,size=(N,N))
C = np.random.randint(low=1,high=5,size=(N,N))
D = np.random.randint(low=1,high=5,size=(N,N))
st = time.time()
A = calc(B,C,D)
et = time.time()
print("Time taken:",et-st)
