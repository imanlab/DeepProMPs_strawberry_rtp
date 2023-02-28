import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn.metrics import mean_squared_error


tfd = tfp.distributions


def get_variables_from_module(module):
    """
    Return a dictionary of all variables in a module
    Source: https://stackoverflow.com/a/28150307
    """
    return {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}


class Log:
    """
    Log class
    """
    def __init__(self, log_dir: str):
        self.datetime_created = datetime.now()
        self.log_dir = log_dir
        # Log dir creation.
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        # Main log file creation.
        self.log_file = os.path.join(log_dir, 'log.txt')
        with open(self.log_file, "w") as f:
            f.write(f"Log file created on {self.datetime_created.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log(self, text: str):
        with open(self.log_file, "a") as f:
            f.write(text)
            f.write("\n")

    def print(self, text: str):
        print(text)
        self.log(text)

    def get_file_path(self, filename: str):
        return os.path.join(self.log_dir, filename)

    def log_config(self, cfg_module):
        config_book = get_variables_from_module(cfg_module)
        with open(self.get_file_path('config.txt'), "w") as f:
            for key, value in config_book.items():
                f.write(f"{key:30}{str(value):100}\n")


def plot_loss(plot_path, loss, name, val_loss=None):
    """
    Plot of the history loss for this training cycle.
    :param plot_path: Save folder of the plot of the loss
    :param loss: Training loss
    :param val_loss: Validation loss
    """
    fig = plt.figure()
    plt.plot(loss, 'r', label='train')
    if val_loss:
        plt.plot(val_loss, 'b', label='val')
    plt.grid(True, which='both')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_path, name))
    plt.close(fig)
    # Same, but logarithmic y scale.
    fig = plt.figure()
    plt.plot(loss, 'r', label='train')
    if val_loss:
        plt.plot(val_loss, 'b', label='val')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_path, 'log_' + name))
    plt.close(fig)


def My_metric(mean_pred, mean_true, cov_pred, cov_true):
    """
    Metric Function
    """
    mse_mean = mean_squared_error(mean_true, mean_pred)
    mse_cov = mean_squared_error(cov_true, cov_pred)

    return mse_cov + 0.5 * mse_mean


def My_metric_final_distance(traj_true, traj_pred):
    """
    Metric Function to calculate the distance between the final predicted and true 3D point.
    final_pose_true => shape(n_dof, test batch, n_t)
    """
    n_test = len(traj_true[0])      # 30
    distance_sum = 0.0

    x_true = traj_true[0][:, -1]    # (30,)
    y_true = traj_true[1][:, -1]
    z_true = traj_true[2][:, -1]

    x_pred = traj_pred[0][:, -1]
    y_pred = traj_pred[1][:, -1]
    z_pred = traj_pred[2][:, -1]

    distance_true_pred = np.sqrt(pow((x_true - x_pred), 2) + pow((y_true - y_pred), 2) + pow((z_true - z_pred), 2))    # (30,)
    for i in range(n_test):
        distance_sum += distance_true_pred[i]
    distance_avg = distance_sum / n_test

    return distance_avg