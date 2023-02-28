import numpy as np
import json

'''
# These functions were used when we did pca on all 7dof
def reconstruct_mean(mean_weights):
    to_norm_eig=mean_weights[0]
    u=mean_weights[1:8]
    v=mean_weights[8:]
    uu=u.reshape((7,1))
    vv = v.reshape((1,8))
    mean_matrix= np.dot(np.dot(uu,to_norm_eig),vv)

    return mean_matrix

def reconstruct_covariance(covariance_weights,n_eig):
    to_norm_eig=covariance_weights[:n_eig]
    u=covariance_weights[n_eig:]
    uu=u.reshape((int(len(u)/n_eig),n_eig))
    ss=np.diag(to_norm_eig)
    covariance_matrix= np.dot(np.dot(uu,ss),uu.T)

    return covariance_matrix
'''


def reconstruct_covariance(covariance_weights):
    s = covariance_weights[0]   # eigenvalue
    u = covariance_weights[1:]  # covariance values
    uu = u.reshape((int(len(u)), 1))
    covariance_matrix = np.dot(np.dot(uu, s), uu.T)

    return covariance_matrix


def reconstruction_precision(cov_real, cov_reconstructed):
    """
    Returns: mean squared error, value-wise, between the real and reconstructed from the PCA covariance matrix
    """
    mse = ((cov_real - cov_reconstructed)**2).mean(axis=None)

    return mse


if __name__ == "__main__":

    # Test the function:
    dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca"
    with open(dataset_dir + "/probabilistic_7dof/000_ConfigStrawberry_mean&covVectors.json", 'r') as fp:
        annotation = json.load(fp)
        mean_weights = np.asarray(annotation[0]['mean_vector']).round(16).astype('float64')
        covariance_weights = np.asarray(annotation[0]['covariance_vector'])
        # the number in square brackets is the dof we are looking at
    covariance_matrix = reconstruct_covariance(covariance_weights)
    fp.close()
