import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from weight_matrix_recomposition import reconstruct_covariance
from ProMP_A import ProMP
from positive_definite import nearestPD, isPD
from scipy.linalg import block_diag


"""
Function to plot the trajectories and compare them. This script is just for testing.
"""

dataset_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca"
n_t = 100
n_basis = 8
n_dof = 1
promp = ProMP(n_basis, n_dof, n_t)


def distribution_from_pca_7dof(dataset_dir):
    """
    Get trajectory'to_norm_eig mean and covariance from pca on 7DOF (TO USE IT IMPORT COMMENTED FUNCTIONS IN weight_matrix_recomposition)
    NOT USED ANYMORE !
    """
    with open(dataset_dir + "/probabilistic/000_ConfigStrawberry_mean&covVectors.json", 'r') as fp:
        annotation = json.load(fp)
        mean_weights = np.asarray(annotation['mean_vector']).round(16).astype('float64')
        covariance_weights = np.asarray(annotation['covariance_vector'])
    mean_matrix = mean_weights      # reconstruct_mean(mean_weights)
    covariance_matrix = reconstruct_covariance(covariance_weights, 4)
    # mean_traj = torch.from_numpy(mean_matrix[0])
    # cov_traj = torch.from_numpy(covariance_matrix[:n_basis,:n_basis])
    mean_traj = tf.convert_to_tensor(mean_matrix)
    cov_traj = tf.convert_to_tensor(covariance_matrix)
    cov_traj = nearestPD(cov_traj)
    # distribution= torch.distributions.multivariate_normal.MultivariateNormal(mean_traj,cov_traj)
    distribution = tfp.distributions.MultivariateNormalTriL(loc=mean_traj, scale_tril=tf.linalg.cholesky(cov_traj))

    return distribution


def distribution_from_pca_on_each_dof(dataset_dir):
    """
    Get trajectory's mean and covariance from pca on each dof, and create probability distribution
    Returns: trajectory probability distribution after applying PCA on the weights on each dof
    """
    with open(dataset_dir + "/probabilistic_7dof_joints/000_ConfigStrawberry_mean&covVectors.json", 'r') as fp:
        annotation = json.load(fp)
        annotation = annotation[5]  # I'm watching the (i+1)st dof
        mean_weights = np.asarray(annotation['mean_vector']).round(16).astype('float64')
        covariance_weights = np.asarray(annotation['covariance_vector'])
    mean_matrix = mean_weights
    print("000 mean:", mean_matrix)
    print("000 cov:", covariance_weights)
    covariance_matrix = reconstruct_covariance(covariance_weights)
    # mean_traj = torch.from_numpy(mean_matrix)
    # cov_traj = torch.from_numpy(covariance_matrix)
    mean_traj = tf.convert_to_tensor(mean_matrix)
    cov_traj = tf.convert_to_tensor(covariance_matrix)
    cov_traj = nearestPD(cov_traj)
    # distributionPCA = torch.distributions.multivariate_normal.MultivariateNormal(mean_traj, cov_traj)
    distributionPCA = tfp.distributions.MultivariateNormalTriL(loc=mean_traj, scale_tril=tf.linalg.cholesky(cov_traj))

    return distributionPCA


def distribution_from_traj_numpy(dataset_dir):
    """
    Get trajectory's weights mean and covariance from trajectory numpy directly, and create probability distribution
    Returns: trajectory probability distribution after retrieving the weights from the trajectory
    """
    weights = []
    for i in range(10):
        Found = True
        try:
            traj = np.load(dataset_dir + "/config0_traj/config0_strawberry0_traj" + str(i) + ".npy")[:, 0]
            # [:, 0] where 0 indicates the dof we are looking at
        except FileNotFoundError:
            Found = False
        if Found:
            weight = promp.weights_from_trajectory(traj, False)  # weights_dir ()
            weights.append(weight)

    weights = np.asarray(weights)
    mean_weights = promp.get_mean_from_weights(weights)
    cov_weights = promp.get_cov_from_weights(weights.T)
    if not(isPD(cov_weights)):
        cov_weights = nearestPD(cov_weights)

    # Create the probability distribution
    weight_mean = tf.convert_to_tensor(mean_weights)
    weight_cov = tf.convert_to_tensor(cov_weights)
    # distributionW = torch.distributions.multivariate_normal.MultivariateNormal(weight_mean, weight_cov)
    distributionW = tfp.distributions.MultivariateNormalTriL(loc=weight_mean, scale_tril=tf.linalg.cholesky(weight_cov))

    return distributionW


def distribution_from_weights(dataset_dir):
    """
    Get trajectory's mean and covariance from weights, and creates probability distribution
    Same as function above but here i load all the dofs and later decide which one to focus on
    """
    weight_mean = []
    for j in range(7):
        weights = []
        for i in range(10):
            Found = True
            try:
                traj = np.load(dataset_dir + "/config0_traj/config0_strawberry0_traj" + str(i) + ".npy")[:, j]
            except FileNotFoundError:
                Found = False
            if Found:
                weight = promp.weights_from_trajectory(traj, False)  # weights_dir ()
                weights.append(weight)

        weights = np.asarray(weights)
        weight_mean.append(promp.get_mean_from_weights(weights))
        wcov = promp.get_cov_from_weights(weights.T)
        if not(isPD(wcov)):
            wcov = nearestPD(wcov)
        if j == 0:
            weight_cov = wcov
        else:
            weight_cov = block_diag(weight_cov, wcov)

    weight_mean = np.asarray(weight_mean)
    weight_mean = tf.convert_to_tensor(weight_mean[0])      # This way I look just at the first dof
    weight_cov = tf.convert_to_tensor(weight_cov[:8, :8])   # This way I look just at the first dof
    # distributionW = torch.distributions.multivariate_normal.MultivariateNormal(weight_mean, weight_cov)
    distributionW = tfp.distributions.MultivariateNormalTriL(loc=weight_mean, scale_tril=tf.linalg.cholesky(weight_cov))

    return distributionW


def distribution_from_prediction(dataset_dir):
    """
    Get trajectory's mean, eig and cov from the model prediction, and create probability distribution
    """
    with open(dataset_dir + "/probabilistic_7dof/219_ConfigStrawberry_mean&covVectors.json", 'r') as fp:
        pred_annotation = json.load(fp)
        pred_annotation = pred_annotation[4]  # I'm watching the (i+1)st dof
        pred_mean_weights = np.asarray(pred_annotation['mean_vector']).round(16).astype('float64')
        pred_covariance_weights = np.asarray(pred_annotation['covariance_vector'])
    pred_mean_matrix = pred_mean_weights
    print("219 mean:", pred_mean_matrix)
    print("219 cov:", pred_covariance_weights)
    pred_covariance_matrix = reconstruct_covariance(pred_covariance_weights)
    # mean_traj = torch.from_numpy(mean_matrix)
    # cov_traj = torch.from_numpy(covariance_matrix)
    pred_mean_traj = tf.convert_to_tensor(pred_mean_matrix)
    pred_cov_traj = tf.convert_to_tensor(pred_covariance_matrix)
    pred_cov_traj = nearestPD(pred_cov_traj)
    # distributionPCA = torch.distributions.multivariate_normal.MultivariateNormal(mean_traj, cov_traj)
    pred_distributionPCA = tfp.distributions.MultivariateNormalTriL(loc=pred_mean_traj, scale_tril=tf.linalg.cholesky(pred_cov_traj))

    return pred_distributionPCA


def plot_from_distribution(distribution, color: str):

    for i in range(10):
        traj = tfp.distributions.Sample(distribution)
        # traj = np.squeeze(traj)
        traj_sample = traj.sample()
        traj_final = promp.trajectory_from_weights(traj_sample.numpy())
        plt.plot(t, traj_final, color)

    return None


if __name__ == "__main__":
    # PLOTS
    t = np.linspace(0, 1, n_t)

    distributionPCA = distribution_from_pca_on_each_dof(dataset_dir)
    # distributionTraj = distribution_from_traj_numpy(dataset_dir)

    plot_from_distribution(distributionPCA, "g")


    """
    #PLOT TRAJECTORIES FROM MEAN AND COV WEIGHT WITHOUT PROMP
    for i in range(10):
        traj = tfp.distributions.Sample(distributionW)
        # traj = np.squeeze(traj)
        traj_sample = traj.sample()
        trajTR = promp.trajectory_from_weights(traj_sample.numpy())
        plt.plot(t, trajTR, 'm')
    
    
    #PLOT TRAJECTORIES FROM PCA MEAN AND COV WEIGHT WITHOUT PROMP
    for i in range(10):
        traj=distribution.rsample(torch.Size([1]))
        traj = np.squeeze(traj)
        trajPCA = promp.trajectory_from_weights(traj.cpu().detach().numpy())
        plt.plot(t, trajPCA, 'y')
    

    # PLOT TRAJECTORIES FROM PCA MEAN AND COV WEIGHT WITHOUT PROMP
    for i in range(10):
        traj = tfp.distributions.Sample(distributionPCA)
        traj_sample = traj.sample()
        trajPCA = promp.trajectory_from_weights(traj_sample.numpy())
        plt.plot(t, trajPCA, 'g')
    """

    """
    # PLOT REAL TRAJECTORY AND TRAJECTORY FROM WEIGHT
    """
    for i in range(10):
        trajT = np.load(dataset_dir + "/config0_traj/config0_strawberry0_traj" + str(i) + "_joints.npy")
        trajT = np.squeeze(trajT)   # Needed just for joint space
        weight = promp.weights_from_trajectory(trajT, False)
        traj_w = promp.trajectory_from_weights(weight)
        tT = np.linspace(0, 1, len(trajT))
        plt.plot(t, traj_w[:, 5], 'b')
        plt.plot(tT, trajT[:, 5], 'r')   # the second term in square brackets indicates the dof


    plt.title("Green: from PCA; Blue: no PCA; Red: GT")
    plt.show()
