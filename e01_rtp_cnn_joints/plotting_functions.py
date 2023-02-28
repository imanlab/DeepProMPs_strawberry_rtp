import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from pathlib import Path

tfd = tfp.distributions


''' Function to plot the multivariate weights distribution '''
def plot_mulivariate_weigths(save_file_path,mean_weights,covariance_weights,n_dof,n_func,show=False,save=False):
    sample_of_weights = np.random.multivariate_normal(mean_weights, covariance_weights, 100).T
    x = np.linspace(1, n_func * n_dof, num=n_func * n_dof)  # (8,0)
    fig,ax = plt.subplots(figsize=(17, 3))
    ax.grid()
    ax.plot(x, sample_of_weights, '.b')
    ax.plot(x, mean_weights, 'or')
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(8)
    ax.set_xlabel('Weights', fontsize=8)
    plt.xticks(x)
    fig.set_dpi(100)
    if show==True:
        plt.show()  # Show the image.
    #Create the path for saving plots.
    if save==True:
        Path(save_file_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(save_file_path + 'weights_distrib_multivariate' + '.png')


''' Function to plot the distribution of one single joint '''


def plot_traj_distribution_1_joint(save_path, config, dof, mean_traj_1, traj_std_1, mean_traj_2=np.zeros(shape=(100,)),
                                   traj_std_2=np.zeros(shape=(100,)), show=False, save=False):

            # 1 distribution (true)
            traj_true_mean = mean_traj_1
            right_traj_1_true = mean_traj_1 + traj_std_1
            left_traj_1_true = mean_traj_1 - traj_std_1
            right_traj_3_true = mean_traj_1 + 3 * traj_std_1
            left_traj_3_true = mean_traj_1 - 3 * traj_std_1

            # 2 distribution (pred)
            if mean_traj_2.all() != 0:
                traj_pred_mean = mean_traj_2
                right_traj_1_pred = mean_traj_2 + traj_std_2
                left_traj_1_pred = mean_traj_2 - traj_std_2
                right_traj_3_pred = mean_traj_2 + 3 * traj_std_2
                left_traj_3_pred = mean_traj_2 - 3 * traj_std_2

            q1true = traj_true_mean[:, [0]]
            q1righttrue = right_traj_3_true[:, [0]]
            q1lefttrue = left_traj_3_true[:, [0]]
            q1righttrue_1 = right_traj_1_true[:, [0]]
            q1lefttrue_1 = left_traj_1_true[:, [0]]

            if mean_traj_2.all() != 0:
                q1pred = traj_pred_mean[:, [0]]
                q1rightpred = right_traj_3_pred[:, [0]]
                q1leftpred = left_traj_3_pred[:, [0]]
                q1rightpred_1 = right_traj_1_pred[:, [0]]
                q1leftpred_1 = left_traj_1_pred[:, [0]]

            fig = plt.figure(figsize=(8, 4))
            fig.suptitle('Trajectories Distributions Configuration  ' + str(config) + ("\nDegree of freedom #")
                         + str(dof+1), fontweight="bold")
            # Degree of Freedom task
            x = np.linspace(0, 100, 100)
            plt.plot(q1true, 'c', label='True', linewidth=0.5)   # plot true mean
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q1righttrue, 'b', linewidth=0.5)   # plot true std, 99%
            plt.plot(q1lefttrue, 'b', linewidth=0.5)
            plt.plot(q1righttrue_1, 'b', linewidth=0.5)     # plot true std, 66%
            plt.plot(q1lefttrue_1, 'b', linewidth=0.5)
            plt.fill_between(x, q1righttrue.reshape(100, ), q1lefttrue.reshape(100, ), alpha=0.25, facecolor='blue')
            plt.fill_between(x, q1righttrue_1.reshape(100, ), q1lefttrue_1.reshape(100, ), alpha=0.25, facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q1pred, 'r', label='Pred', linewidth=0.5)  # plot pred mean
                plt.plot(q1rightpred, 'm', linewidth=0.5)   # plot pred std, 99%
                plt.plot(q1leftpred, 'm', linewidth=0.5)
                plt.plot(q1rightpred_1, 'm', linewidth=0.5)     # plot pred std, 66%
                plt.plot(q1leftpred_1, 'm', linewidth=0.5)
                plt.fill_between(x, q1rightpred.reshape(100, ), q1leftpred.reshape(100, ),
                                 alpha=0.25, facecolor=(1, 0, 0, .4))
                plt.fill_between(x, q1rightpred_1.reshape(100, ), q1leftpred_1.reshape(100, ),
                                 alpha=0.25, facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')

            fig.set_dpi(200)
            if show == True:
                plt.show()  # Show the image.
            if save == True:
                # Create the path for saving plots.
                Path(save_path).mkdir(exist_ok=True, parents=True)
                plt.savefig(save_path+'/task_traj_distrib_' + str(config) + "_dof" + str(dof+1) + '.png')


# Function to plot the weights distribution of one joint
def plot_weights_distributions_1_joint(save_file_path , mean_weights_1, std_weights_1, n_func, mean_weights_2=np.zeros(shape=(8,)), std_weights_2=np.zeros(shape=(8,)), show=True, save=False):
    x = np.linspace(1, n_func, num=n_func)  # (8,0)
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Weights distributions configuration  ', fontweight="bold", fontsize=7)
    plt.bar(x, mean_weights_1, yerr=std_weights_1, align='center', alpha=0.5, ecolor='black', capsize=5)
    if mean_weights_2.all()!= 0 and std_weights_2.all()!= 0:
         plt.bar(x, mean_weights_2, yerr=std_weights_2,align='center', alpha=0.8, ecolor='red', color=(1, 0, 0, .4), capsize=5)
    fig.set_dpi(200)
    if show == True:
        plt.show()  # Show the image.
    # Create the path for saving plots.
    if save == True:
        Path().mkdir(exist_ok=True, parents=True)
        plt.savefig(save_file_path + 'weights_distrib' + '.png')