import matplotlib.pyplot as plt
import numpy as np

from formation_distance import formation_distance
# Formation Control
# Shape

side = 8
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
beta = 0.5
G = 1e-1
fc = formation_distance(2, 1, d_vector, mu, tilde_mu, B_matrix, G/beta, G)
np.random.seed(2)
# Simulation parameters
T = 1000
time = range(T)

frame_every = 1
tracks = np.zeros([T//frame_every, 2*N])
X = np.random.uniform(-10,10,2*N)
V = np.zeros(2*N)
for t in time:
    U = fc.u_acc(X, V)
    # print('X', X)
    # print('V', V)
    # print('U', U)
    V += U/3 #+ np.random.randn(2*N)/50

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
plt.legend()
plt.axis('equal')

plt.show(block=True)
