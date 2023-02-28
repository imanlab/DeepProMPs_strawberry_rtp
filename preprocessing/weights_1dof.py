import numpy as np
import json

from ProMP_A import ProMP
from matplotlib import pyplot as plt

n_t = 100
n_basis = 8
n_dof = 1
promp = ProMP(n_basis, n_dof, n_t)
dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca"  # "/home/chiara/Deep_movement_planning/dataset"

if __name__ == "__main__":

    '''Load all trajectories for each dof (for all strawberries in all configurations)'''
    hlist = []
    hlistM = []
    for j in range(5):
        for k in range(20):
            mean_cov_vector = []
            for d in range(7):
                weights = []
                for i in range(10):
                    Found = True
                    try:
                        # take the trajectory of the d-th dof. Squeeze needed only for joints
                        traj = np.squeeze(np.load(dataset_dir + "/config" + str(j) + "_traj/config" + str(j) +
                                                  "_strawberry" + str(k) + "_traj" + str(i) + "_joints.npy"))[:, d]
                    except FileNotFoundError:
                        Found = False
                    if Found:
                        weight = promp.weights_from_trajectory(traj, False)  # weights_dir (7,8)
                        weights.append(weight)

                '''Extract mean and covariance from weights_dir of the 10 trajectories'''
                weights = np.asarray(weights)
                if weights.size > 0:
                    weight_cov = promp.get_cov_from_weights(weights.T)
                    weight_mean = promp.get_mean_from_weights(weights)

                    u, s, vh = np.linalg.svd(weight_cov)  # apply Singular Value Decomposition

                    '''save matrices with pca'''
                    s_pca = s[0]
                    u_pca = u[:, 0]
                    covariance_vector = np.append(s_pca, u_pca)

                jsonlist = {"mean_vector": weight_mean.tolist(), "covariance_vector": covariance_vector.tolist()}
                mean_cov_vector.append(jsonlist)

            if k > 9:  # create ID: config number + strawberry number
                kk = str(k)
            else:
                kk = "0" + str(k)
            filepath = dataset_dir + "/probabilistic_7dof_joints/" + str(j) + kk + "_ConfigStrawberry_mean&covVectors.json"
            with open(filepath, 'w') as f:
                json.dump(mean_cov_vector, f)

                '''pca preparation
                tot = 0
                for z in range(len(to_norm_eig)):
                    tot += to_norm_eig[z]
                EigenvaluesNorm = 100* to_norm_eig / tot
                l = []
                l.append(EigenvaluesNorm)
                l.append("c" + str(j) + "to_norm_eig" + str(k))
                hlist.append(l)'''


    ''' HOW MANY EIGENVALUES CAN WE CHOOSE? (1)
    cov_eig=[]
    for k in range(len(EigenvaluesNorm)):
        cov_eig_sum = 0
        for i in range(len(hlist)):
            hlist_sum = 0
            for j in range(k+1):
                try:
                    hlist_sum += hlist[i][0][j]
                except IndexError:
                    print(hlist[i][1])
            cov_eig_sum += hlist_sum
        cov_eig.append(cov_eig_sum/95)
    
    plt.bar(range(len(EigenvaluesNorm)), cov_eig)
    plt.show()'''


    """ 
    CHECK THAT U AND V ARE THE SAME BECAUSE COV IS SYMMETRIC POSITIVE DEFINITE
    ==> THEY ARE THE SAME CONVERTING u and vh TO DTYPE FLOAT16
    
    same = False
    stay = True
    for i in range(len(u)):
        if stay:
            for j in range(len(u)):
                if stay:
                    if u[i][j]==vh.T[i][j]:
                        same=True
                        #print("u:", u[i][j], "v:", vh[i][j])
                        #print(same)
                    else:
                        same= False
                        stay = False
                        #print("u:", u[i][j], "v:", vh[i][j])
    print(same)
    """

