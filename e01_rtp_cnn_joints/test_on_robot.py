import os
import numpy as np
import tensorflow as tf
import json
import config_joints as cfg
import tensorflow_probability as tfp
import tensorflow_probability as tfp

from skimage.io import imread
from skimage.transform import resize
from numpy import linalg as la
from utils.models import get_encoder_model
from config_joints import *
from losses import RMSE
from ProMP import ProMP
from weight_matrix_recomposition import reconstruct_covariance_tensor


tf.data.experimental.enable_debug_mode()


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico'to_norm_eig `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3) and np.linalg.det(A3) != 0:
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3) or np.linalg.det(A3) == 0:
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def load_image(img_path, resize_shape=(256, 256, 3)):
    """
    Loads the original RGB image
    """
    img = imread(img_path, pilmode='RGB')
    if resize_shape:
        img = resize(img, resize_shape)
    return img.astype('float32')


if __name__ == "__main__":

    print("Preparing data and loading models..")
    # Load the ProMP
    promp = ProMP(8, 1, 100)

    # Load the Autoencoder model and extract the Encoder model
    encoder = get_encoder_model(tf.keras.models.load_model(ENCODER_MODEL_PATH))

    # Load the MLP model
    loss = RMSE(promp)
    load_model_name = "model_21_02__15_08"
    model_load_path = os.path.join(MODEL_FOLDER, load_model_name)
    model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})

    # Load the data
    input_img = "/home/francesco/PycharmProjects/deep_movement_primitives-main/images/proper_robot_test_2.png"
    latent = encoder(np.expand_dims(load_image(input_img), axis=0))
    encoded_img = np.squeeze(latent)
    print("Done!")

    # Prediction
    pred = model.predict(latent)
    predicted_mean = pred[:, 0:56]  # (1, 56)
    predicted_pca = pred[:, -63:]   # (1, 63)
    MEAN_TRAJ_pred_dofs = []
    COV_TRAJ_pred_dofs = []
    pdfs = []
    traj_dofs = []
    for dof in range(7):
        mean_pred_dof = predicted_mean[:, (dof * 8):((dof + 1) * 8)]  # (1, 8)
        mean_pred_dof = np.squeeze(mean_pred_dof)       # (8,)
        pca_pred_dof = predicted_pca[:, (dof * 9):((dof + 1) * 9)]  # (1, 9)
        pca_pred_dof = np.squeeze(pca_pred_dof)

        cov_pred_dof = np.asarray(reconstruct_covariance_tensor(pca_pred_dof))  # (8, 8)
        is_pos_def_pred = isPD(cov_pred_dof)  # The number in square brackets selects the dof
        print('The predicted weights covariance matrix is positive definite?   ', is_pos_def_pred)
        mean_pred_dof_traj = tf.convert_to_tensor(mean_pred_dof)
        cov_pred_dof_traj = tf.convert_to_tensor(cov_pred_dof)
        cov_pred_dof_traj = nearestPD(cov_pred_dof_traj)

        # MEAN_TRAJ_pred = promp.trajectory_from_weights(mean_pred_dof[0], vector_output=False)  # (100, 1)
        # COV_TRAJ_pred = promp.get_traj_cov(cov_pred_dof[0]).astype('float64')  # (100, 100)
        # STD_TRAJ_pred = promp.get_std_from_covariance(COV_TRAJ_pred)
        # STD_TRAJ_pred = np.reshape(STD_TRAJ_pred, (100, -1), order='F')  # (100, 1)
        #
        # MEAN_TRAJ_pred_dofs.append(MEAN_TRAJ_pred)
        # COV_TRAJ_pred_dofs.append(COV_TRAJ_pred)

        pdf = tfp.distributions.MultivariateNormalTriL(loc=mean_pred_dof_traj,
                                                       scale_tril=tf.linalg.cholesky(cov_pred_dof_traj))
        # pdfs.append(pdf)
        traj = tfp.distributions.Sample(pdf)
        traj_sample = traj.sample()
        traj_final = promp.trajectory_from_weights(traj_sample)
        traj_dofs.append(traj_final)

    traj_dofs = np.asarray(traj_dofs, dtype="float32")
    np.save("/home/francesco/PycharmProjects/deep_movement_primitives-main/predictions/model_21_02__15_08/proper_robot_test3.npy",
            traj_dofs)
