import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from weight_matrix_recomposition import reconstruct_covariance_tensor_batch

tfd = tfp.distributions

# np.set_printoptions(threshold=np.inf)
tf.config.run_functions_eagerly(True)


def RMSE(promp):
    all_phi = tf.cast(promp.all_phi(), 'float64')

    def RMSE_loss_concatenated(ytrue, ypred):

        mean_traj_true_dofs = []
        cov_traj_true_dofs = []
        mean_traj_pred_dofs = []
        cov_traj_pred_dofs = []

        mean_true = ytrue[:, 0:56]
        pca_true = ytrue[:, -63:]
        mean_pred = ypred[:, 0:56]
        pca_pred = ypred[:, -63:]

        # Get the process for each degree of freedom
        for i in range(7):
            mean_true_dof = mean_true[:, (8 * i):(8 * (i + 1))]  # (training batch, 8)
            # mean_true_dofs.append(mean_true_dof)      Saves the list of vectors
            mean_traj_true_dof = tf.transpose(
                tf.matmul(all_phi, tf.transpose(mean_true_dof)))  # to traj domain. (training batch, N_t)
            mean_traj_true_dofs.append(mean_traj_true_dof)

            pca_true_dof = pca_true[:, (9 * i):(9 * (i + 1))]
            # pca_true_dofs.append(pca_true_dof)        Saves the list of vectors
            cov_true_dof = reconstruct_covariance_tensor_batch(pca_true_dof)  # from pca to reconstructed cov matrix (8x8)
            cov_traj_true_dof = tf.linalg.matmul(cov_true_dof, tf.transpose(all_phi))
            cov_traj_true_dof = tf.linalg.matmul(tf.transpose(cov_traj_true_dof, perm=[0, 2, 1]),
                                                 tf.transpose(all_phi))  # to traj domain. shape (batch, N_t, N_t)
            cov_traj_true_dofs.append(cov_traj_true_dof)

            mean_pred_dof = mean_pred[:, (8 * i):(8 * (i + 1))]
            # mean_pred_dofs.append(mean_pred_dof)      Saves the list of vectors
            mean_traj_pred_dof = tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred_dof)))  # to traj domain
            mean_traj_pred_dofs.append(mean_traj_pred_dof)

            pca_pred_dof = pca_pred[:, (9 * i):(9 * (i + 1))]
            # pca_pred_dofs.append(pca_pred_dof)        Saves the list of vectors
            cov_pred_dof = reconstruct_covariance_tensor_batch(pca_pred_dof)  # from pca to reconstructed cov matrix (8x8)
            cov_traj_pred_dof = tf.linalg.matmul(cov_pred_dof, tf.transpose(all_phi))
            cov_traj_pred_dof = tf.linalg.matmul(tf.transpose(cov_traj_pred_dof, perm=[0, 2, 1]),
                                                 tf.transpose(all_phi))  # to traj domain
            cov_traj_pred_dofs.append(cov_traj_pred_dof)

        mean_traj_true_dofs = tf.convert_to_tensor(mean_traj_true_dofs)  # Convert to tensor. shape = (dofs, batch, N_t)
        mean_traj_true_dofs = tf.transpose(mean_traj_true_dofs, perm=[1, 0, 2])  # (batch, dofs, N_t)

        cov_traj_true_dofs = tf.convert_to_tensor(
            cov_traj_true_dofs)  # Convert to tensor. shape = (dofs, batch, N_t, N_t)
        cov_traj_true_dofs = tf.transpose(cov_traj_true_dofs, perm=[1, 0, 2, 3])  # (batch, dofs, N_t, N_t)

        mean_traj_pred_dofs = tf.convert_to_tensor(mean_traj_pred_dofs)
        mean_traj_pred_dofs = tf.transpose(mean_traj_pred_dofs, perm=[1, 0, 2])

        cov_traj_pred_dofs = tf.convert_to_tensor(cov_traj_pred_dofs)
        cov_traj_pred_dofs = tf.transpose(cov_traj_pred_dofs, perm=[1, 0, 2, 3])

        loss_mean = (K.mean(K.square(mean_traj_true_dofs - mean_traj_pred_dofs)))
        loss_cov = (K.mean(K.square(cov_traj_true_dofs - cov_traj_pred_dofs)))
        return 0.1 * loss_mean + loss_cov

    def RMSE_loss_single(ytrue, ypred):

        mean_true = ytrue[:, 0:56]
        pca_true = ytrue[:, -63:]
        mean_pred = ypred[:, 0:56]
        pca_pred = ypred[:, -63:]
        loss_dofs = []

        for i in range(7):
            mean_true_dof = mean_true[:, (8 * i):(8 * (i + 1))]
            pca_true_dof = pca_true[:, (9 * i):(9 * (i + 1))]
            mean_pred_dof = mean_pred[:, (8 * i):(8 * (i + 1))]
            pca_pred_dof = pca_pred[:, (9 * i):(9 * (i + 1))]

            mean_traj_true_dof = tf.transpose(tf.matmul(all_phi, tf.transpose(mean_true_dof)))
            mean_traj_pred_dof = tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred_dof)))

            cov_true_dof = reconstruct_covariance_tensor_batch(pca_true_dof)
            cov_pred_dof = reconstruct_covariance_tensor_batch(pca_pred_dof)

            cov_traj_true_dof = tf.linalg.matmul(cov_true_dof, tf.transpose(all_phi))
            cov_traj_true_dof = tf.linalg.matmul(tf.transpose(cov_traj_true_dof, perm=[0, 2, 1]), tf.transpose(all_phi))
            cov_traj_pred_dof = tf.linalg.matmul(cov_pred_dof, tf.transpose(all_phi))
            cov_traj_pred_dof = tf.linalg.matmul(tf.transpose(cov_traj_pred_dof, perm=[0, 2, 1]), tf.transpose(all_phi))

            loss_mean = (K.mean(K.square(mean_traj_true_dof - mean_traj_pred_dof)))
            loss_cov = (K.mean(K.square(cov_traj_true_dof - cov_traj_pred_dof)))
            loss_dof = 0.1 * loss_mean + loss_cov
            loss_dofs.append(loss_dof)

        for j in range(len(loss_dofs)):
            if j == 0:
                loss_tot = loss_dofs[j]
            else:
                loss_tot = loss_tot + loss_dofs[j]

        return loss_tot

    return RMSE_loss_single
