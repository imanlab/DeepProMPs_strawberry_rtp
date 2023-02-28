import os
import numpy as np
import tensorflow as tf
import json
import config_task as cfg
import tensorflow_probability as tfp

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config
from plotting_functions import plot_traj_distribution_1_joint
from datasets import Dataset_RGBProMP
from experiments import Experiment
from losses import RMSE

from output import plot_loss, My_metric, My_metric_final_distance
from ProMP import ProMP
from pathlib import Path
from utils.models import get_encoder_model, get_convolutional_model
from weight_matrix_recomposition import *
from numpy import linalg as la


tfd = tfp.distributions
np_config.enable_numpy_behavior()
tf.config.run_functions_eagerly(True)
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
    # The above is different from [1]. It appears that MATLAB'to_norm_eig `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy'to_norm_eig will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'to_norm_eig `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3) or np.linalg.det(A3) == 0:
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def nearestPD_batch(A_batch):
    """
    A_batch: batch of covariance matrices to turn into positive definite matrices. Shape (batch size, N, N).
    """
    for i in range(len(A_batch)):
        A_batch[i] = nearestPD(A_batch[i])

    return A_batch


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# def is_pos_def(x, tol=0):
#     """
#     Function that defines if a matrix is positive definite
#     """
#     return np.all(np.linalg.eigvals(x) > tol)


class Experiment_ProMPs(Experiment):
    """
    Experiment Class with train and test functions.
    CNN model predicting full ProMP weights of RTP using the RTP-RGB dataset.

    Input: Images from home position

    Output: full ProMP weights of RTP trajectory

    Model: RGB image -> Encoder -> bottleneck image -> CNN + FC -> ProMP_weights

    Training loss: RMSE on mean and covariance (check)
    """
    def __init__(self):
        super().__init__(cfg)

        ''' ProMPs Variables '''
        self.N_BASIS = 8
        self.N_DOF = 1
        self.N_T = 100
        self.promp = ProMP(self.N_BASIS, self.N_DOF, self.N_T)
        # Load the autoencoder model and extract the encoder model
        encoder = get_encoder_model(tf.keras.models.load_model(self.cfg.ENCODER_MODEL_PATH))

        # Load the dataset
        print("Loading data...")
        self.dataset = Dataset_RGBProMP(encoder=encoder, dataset_dir=cfg.ANNOTATION_PATH, rgb_dir=cfg.IMAGE_PATH)
        self.dataset.prepare_data()
        print("Done!")

        # Load the model
        self.model = get_convolutional_model(output_size_mean=56, output_size_pca=63, activation="relu",
                                             l1_reg=float(0), l2_reg=float(0), name="deep_ProMPs_model_RGB")
        # Choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

        # Select loss function
        self.loss = RMSE(self.promp)

        # Callbacks.
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=cfg.es["delta"], patience=cfg.es["patience"],
                                       verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=cfg.rl["factor"], patience=cfg.rl["patience"],
                                      min_lr=cfg.rl["min_lr"])
        self.callbacks = []
        if cfg.es["enable"]:
            self.callbacks.append(early_stopping)
        if cfg.rl["enable"]:
            self.callbacks.append(reduce_lr)

    def train(self):
        if self.callbacks is None:
            self.callbacks = []

        ''' Load the data '''
        (X_train, y_train), (X_val, y_val), (_, _) = self.dataset.data
        encoded_train = X_train["encoded"]
        encoded_val = X_val["encoded"]
        mean_train = np.asarray(y_train['mean_weights'])
        cov_train = y_train["cov_weights"]
        mean_val = y_val['mean_weights']
        cov_val = y_val["cov_weights"]
        yt = np.hstack((mean_train, cov_train))
        yv = np.hstack((mean_val, cov_val))

        print('encoded train:   ', np.shape(encoded_train))
        print('mean train:   ', mean_train.shape)
        print('cov train:   ', cov_train.shape)
        print('yt train:   ', yt.shape)
        print('encoded val:   ', np.shape(encoded_val))
        print('mean val:   ', mean_val.shape)
        print('cov val:   ', cov_val.shape)
        print('yt val:   ', yv.shape)

        '''Load the models'''
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        '''Train '''
        history = self.model.fit(encoded_train, yt, epochs=self.cfg.epochs, batch_size=self.cfg.batch,
                                 validation_data=(encoded_val, yv), callbacks=self.callbacks)

        '''Save the model '''
        Path(self.cfg.MODEL_PATH).mkdir(exist_ok=True, parents=True)
        Path(self.cfg.LOSS_PATH).mkdir(exist_ok=True, parents=True)
        self.model.save(self.cfg.MODEL_PATH)

        ''' Save plot of training loss '''
        plot_loss(self.cfg.LOSS_PATH, history.history['loss'], name='loss.png', val_loss=history.history['val_loss'])

    def eval(self, load_model_name):
        loss = self.loss
        ''' Load the model'''
        model_load_path = os.path.join(self.cfg.MODEL_FOLDER, load_model_name)
        model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})

        ''' Load the data '''
        (_, _), (_, _), (X_true, y_true) = self.dataset.data
        encoded_true = X_true["encoded"]

        ''' Predict'''
        ypred = model.predict(encoded_true)
        print('Tested configurarions:', self.dataset.data_names['test_ids'])
        predicted_mean = ypred[:, 0:56]
        predicted_pca = ypred[:, -63:]
        mean_true_dofs = []
        pca_true_dofs = []
        mean_pred_dofs = []
        pca_pred_dofs = []
        cov_true_dofs = []
        cov_pred_dofs = []
        traj_true_dofs = []
        traj_pred_dofs = []
        for dof in range(7):
            mean_true_dof = np.asarray(y_true['mean_weights'][:, (dof * 8):((dof + 1) * 8)]).astype('float64')   # (test batch, 8)
            pca_true_dof = np.asarray(y_true['cov_weights'][:, (dof * 9):((dof + 1) * 9)]).astype('float64')     # (test batch, 9)
            mean_true_dofs.append(mean_true_dof)
            pca_true_dofs.append(pca_true_dof)

            mean_pred_dof = predicted_mean[:, (dof * 8):((dof + 1) * 8)]    # (test batch, 8)
            pca_pred_dof = predicted_pca[:, (dof * 9):((dof + 1) * 9)]      # (test batch, 9)
            mean_pred_dofs.append(mean_pred_dof)
            pca_pred_dofs.append(pca_pred_dof)

            cov_true_dof = np.asarray(reconstruct_covariance_np_batch(pca_true_dof))    # (test batch, 8, 8)
            cov_true_dof = nearestPD_batch(cov_true_dof)    #
            cov_pred_dof = np.asarray(reconstruct_covariance_np_batch(pca_pred_dof))    # (test batch, 8, 8)
            cov_pred_dof = nearestPD_batch(cov_pred_dof)    #
            cov_true_dofs.append(cov_true_dof)
            cov_pred_dofs.append(cov_pred_dof)

            ''' Next part is for My_metric_final_distance '''
            mean_true_dof_traj = tf.convert_to_tensor(mean_true_dof)
            # cov_true_dof_traj = nearestPD_batch(cov_true_dof)
            cov_true_dof_traj = tf.convert_to_tensor(cov_true_dof)
            pdf_true = tfp.distributions.MultivariateNormalTriL(loc=mean_true_dof_traj,
                                                                scale_tril=tf.linalg.cholesky(cov_true_dof_traj))   # (batch_shape=30, event_shape=8)
            traj_true_dof = (self.promp.trajectory_from_weights(tfp.distributions.Sample(pdf_true).sample())).T     # (100, test batch)

            mean_pred_dof_traj = tf.convert_to_tensor(mean_pred_dof)
            # cov_pred_dof_traj = nearestPD_batch(cov_pred_dof)
            cov_pred_dof_traj = tf.convert_to_tensor(cov_pred_dof)
            pdf_pred = tfp.distributions.MultivariateNormalTriL(loc=mean_pred_dof_traj,
                                                                scale_tril=tf.linalg.cholesky(cov_pred_dof_traj))
            traj_pred_dof = (self.promp.trajectory_from_weights(tfp.distributions.Sample(pdf_pred).sample())).T

            traj_true_dofs.append(traj_true_dof)      # transposing traj before appending -> (test batch, 100)
            traj_pred_dofs.append(traj_pred_dof)

        # Uncomment the next one if you want to save the final_distance_metric
        final_distance_metric = My_metric_final_distance(traj_true=traj_true_dofs, traj_pred=traj_pred_dofs)

        is_pos_def_pred = isPD(cov_pred_dofs[0])    # The number in square brackets selects the dof
        is_pos_def_true = isPD(cov_true_dofs[0])    # The number in square brackets selects the dof
        print('The predicted weights covariance matrix is positive definite?   ', is_pos_def_pred)
        print('The true weights covariance matrix is positive definite?   ', is_pos_def_true)

        ''' Compute the metric for each dof. Plotting the true and predicted probabilistic distributions '''
        metric_dofs = []
        metric_tot = 0.0
        previous_id = 100    # symbolic. Just a ID we don't have in the dataset
        for dof in range(7):
            n_test = cov_pred_dofs[dof].shape[0]      # retrieve the number of test samples (30)
            metric = 0.0
            # Scrolls the dof to test
            mean_true_totest = mean_true_dofs[dof]
            cov_true_totest = cov_true_dofs[dof]
            mean_pred_totest = mean_pred_dofs[dof]
            cov_pred_totest = cov_pred_dofs[dof]
            for i in range(n_test):
                current_id = int((self.dataset.data_names['test_ids'][i]).split(sep='/')[-1].strip('.png').split(sep='_')[0])
                MEAN_TRAJ_true = self.promp.trajectory_from_weights(mean_true_totest[i, :], vector_output=False)    # (100, 1)
                MEAN_TRAJ_pred = self.promp.trajectory_from_weights(mean_pred_totest[i, :], vector_output=False)    # (100, 1)
                COV_TRAJ_true = self.promp.get_traj_cov(cov_true_totest[i, :, :]).astype('float64')     # (100, 100)
                STD_TRAJ_true = self.promp.get_std_from_covariance(COV_TRAJ_true)
                STD_TRAJ_true = np.reshape(STD_TRAJ_true, (100, -1), order='F')     # (100, 1)
                COV_TRAJ_pred = self.promp.get_traj_cov(cov_pred_totest[i, :, :]).astype('float64')     # (100, 100)
                STD_TRAJ_pred = self.promp.get_std_from_covariance(COV_TRAJ_pred)
                STD_TRAJ_pred = np.reshape(STD_TRAJ_pred, (100, -1), order='F')     # (100, 1)
                metric += My_metric(MEAN_TRAJ_pred, MEAN_TRAJ_true, COV_TRAJ_pred, COV_TRAJ_true)
                metric_tot += My_metric(MEAN_TRAJ_pred, MEAN_TRAJ_true, COV_TRAJ_pred, COV_TRAJ_true)

                ''' Plot the predictions '''
                if current_id != previous_id:
                    plot_traj_distribution_1_joint(save_path=os.path.join(self.cfg.OUTPUT_PATH, load_model_name),
                                                   config=self.dataset.data_names['test_ids'][i], dof=dof,
                                                   mean_traj_1=MEAN_TRAJ_true, traj_std_1=STD_TRAJ_true,
                                                   mean_traj_2=MEAN_TRAJ_pred, traj_std_2=STD_TRAJ_pred,
                                                   show=False, save=True)

                previous_id = int((self.dataset.data_names['test_ids'][i]).split(sep='/')[-1].strip('.png').split(sep='_')[0])    # used to not plot too much figures
            # This way we are saving a metric for each dof. Create also one that takes the whole metric
            metric = metric / n_test
            metric_dofs.append(metric)

        ''' Save the metric, both for each dof and total '''
        annotation = {}
        for j in range(len(metric_dofs)):
            annotation["RMSE_dof_" + str(j+1)] = str(metric_dofs[j])
            print('The average RMSE on single dof is:  ', annotation["RMSE_dof_" + str(j+1)])

        metric_tot = metric_tot / (n_test * 7)
        # Save the metric total metric, across al samples, across all dofs
        annotation["RMSE_TOT"] = str(metric_tot)
        print('The average total RMSE is:  ', annotation["RMSE_TOT"])
        # Save the final_distance_metric
        annotation["AVERAGE DISTANCE FROM FINAL POINT (m)"] = final_distance_metric
        print("The average distance from the final point is (m):", final_distance_metric)
        dump_file_path = os.path.join(self.cfg.METRIC_PATH, load_model_name) + '/metric.json'
        Path(os.path.join(self.cfg.METRIC_PATH, load_model_name)).mkdir(exist_ok=True, parents=True)
        with open(dump_file_path, 'w') as f:
            json.dump(annotation, f)


if __name__ == "__main__":
    '''
    DEFINE THE EXPERIMENT
    '''
    Experiment_ProMPs = Experiment_ProMPs()
    '''
    TRAIN THE MODEL
    '''
    # Experiment_ProMPs.train()
    '''
    TEST THE MODEL
    '''
    Experiment_ProMPs.eval(load_model_name='model_17_02__14_37')

