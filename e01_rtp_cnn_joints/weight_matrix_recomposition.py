import numpy as np
import json
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def reconstruct_covariance_np(covariance_weights):
    s = covariance_weights[0]   # eigenvalue
    u = covariance_weights[1:]  # covariance values
    uu = u.reshape((int(len(u)), 1))
    covariance_matrix = np.dot(np.dot(uu, s), uu.T)      # numpy version

    return covariance_matrix


def reconstruct_covariance_np_batch(covariance_weights_batch):
    """
    Args:
        covariance_weights_batch: array of shape (training batch, 9) containing the pca covariance weights

    Returns: list (MAY NEED TO CONVERT TO ARRAY) of shape (training batch, N_basis, N_basis) containing the reconstructed (post pca) covariance matrices

    """
    covariance_matrix_batch = []
    for i in range(len(covariance_weights_batch)):
        covariance_weight = covariance_weights_batch[i]
        covariance_matrix = reconstruct_covariance_np(covariance_weight)
        covariance_matrix_batch.append(covariance_matrix)

    return covariance_matrix_batch


def reconstruct_covariance_tensor(covariance_weights):
    s = covariance_weights[0]   # eigenvalue
    u = covariance_weights[1:]  # covariance values
    uu = u.reshape((int(len(u)), 1))
    # covariance_matrix = np.dot(np.dot(uu, s), uu.T)       => numpy version
    covariance_matrix = tf.tensordot(tf.tensordot(uu, s, axes=0), uu.T, axes=1)

    return covariance_matrix


def reconstruct_covariance_tensor_batch(covariance_weights_batch):
    """
    Args:
        covariance_weights_batch: array of shape (training batch, 9) containing the pca covariance weights

    Returns: list (MAY NEED TO CONVERT TO ARRAY) of shape (training batch, N_basis, N_basis) containing the reconstructed (post pca) covariance matrices

    """
    covariance_matrix_batch = []
    for i in range(len(covariance_weights_batch)):
        covariance_weight = covariance_weights_batch[i]
        covariance_matrix = reconstruct_covariance_tensor(covariance_weight)
        covariance_matrix_batch.append(covariance_matrix)

    return covariance_matrix_batch


def reconstruction_precision(cov_real, cov_reconstructed):
    """
    Returns: mean squared error, value-wise, between the real and reconstructed from the PCA covariance matrix
    """
    mse = ((cov_real - cov_reconstructed)**2).mean(axis=None)

    return mse


if __name__ == "__main__":

    # Test the function:
    dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_autoencoder"
    with open(dataset_dir + "/probabilistic_renamed/50.json", 'r') as fp:
        annotation = json.load(fp)
        mean_weights = np.asarray(annotation[0]['mean_vector']).round(16).astype('float64')
        covariance_weights = np.asarray(annotation[5]['covariance_vector'])
        # the number in square brackets is the dof we are looking at
    covariance_matrix = reconstruct_covariance_np(covariance_weights)
    fp.close()
