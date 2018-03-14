import matplotlib.pyplot as plt
import numpy as np

from formation_distance import formation_distance
# Formation Control
# Shape

side = 50
N=3
if N==3:
    B_matrix = np.array([[1, 0, -1],
                         [-1, 1, 0],
                         [0, -1, 1]])
    d_vector = np.array([side, side, side])

    mu = np.ones(3)*0
    tilde_mu = np.ones(3)*0
elif N==4:
    # 0     1
    #
    # 2     3
    B_matrix = np.array([[1,  0, -1,  0, -1],
                         [-1, 1,  0,  0,  0],
                         [0, -1,  1, -1,  0],
                         [0,  0,  0,  1,  1]])
    d_vector = np.array([side, side, side*np.sqrt(2), side, side])
    # Motion (see page55)
    k=1e-4
    mu = k*np.array([-5, 0, 0, 0, 0])
    tilde_mu = k*np.array([5, 0, 0, 0, 0])
c_shape = 0.1
c_vel = 0.05
fc = formation_distance(2, 1, d_vector, mu, tilde_mu, B_matrix, c_shape=c_shape, c_vel=c_vel)
np.random.seed(5)
# Simulation parameters
T = 200
time = range(T)

frame_every = 1
tracks = np.zeros([T//frame_every, 2*N])
X = np.random.uniform(-5,5,2*N)
V = np.zeros(2*N)
for t in time:
    U = fc.u_vel(X)
    # print('X', X)
    # print('V', V)
    # print('U', U)
    V += U #+ np.random.randn(2*N)/50

    for d in range(N):
        if not np.allclose(V[d:d+2], np.zeros(2)):
            V[d:d+2] -=  V[d:d+2]* 0.01 * np.linalg.norm(V[d:d+2])
    # vel = fc.u_vel(X)
    # print(vel)
    X += V
    if t % frame_every == 0:
        tracks[t//frame_every, :] = X

plt.figure()
for d in range(N):
    plt.plot(tracks[:,d],tracks[:,d+1],':',linewidth=2, label=f'drone {d+1}')
    plt.plot(tracks[-1, d], tracks[-1, d + 1], '.k', linewidth=2, label=None)


for i in range(B_matrix.shape[1]):
    c = B_matrix[:,i]
    i1 = np.argwhere(c == -1).flatten()[0]
    i2 = np.argwhere(c == 1).flatten()[0]

    dv=X[i1:i1+2] - X[i2:i2+2]
    d = np.linalg.norm(dv)
    print(f"Edge from {i1} to {i2} is {d} units")

    plt.plot(X[[i1,i2]], X[[i1+1,i2+1]], "--k", linewidth=1,label=None)
plt.legend()
plt.axis('equal')
plt.show(block=True)
