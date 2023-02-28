# DeepProMPs_strawberry_rtp

## About The Project 

The aim of the project is to apply the deep probabilistic movement primitives framework in the Agri-robotics filed, to solve the reach to pick task for a strawberry harvest application. 

The action of approaching the strawberry is performed by a Franka Emika robotic arm, equipped with an eye-on-hand RealSense camera. The camera takes an RGB image of the strawberry cluster, which is going to be the input of our deep learning model. 

The architecture of the model is composed of a CNN-Autoencoder, from which we retrieve the latente space of the strawberry cluster image, and a MultiLayerPerceptron, from which we predict the trajectories for the robotic arm to execute. The latent space is used as input for the MLP, that is made of three convolutional layers and two dense layer. 

We trained the MLP model on ProMPs weights, retrieved from a set of trajectories collected by human-expertise. We used a probabilistic approach, defining a trajectory probability distribution of weights, defined by its mean and covariance. This operations where performed both in task and joint space.

You can see a picture of the proposed architecture below:

## What's inside 

### Working environment

Models were built and trained using a conda environment with Python 3.9 and Tensorflow 2.7. You can replicate the same virtual environment from the "environment.yml" with the following command:

        conda env create -f environment.yaml -n "NAME_OF_THE_ENV"

### Dataset

The complete raw dataset can be found at this link, where you will also find additional useful informations: https://github.com/imanlab/Franka_datacollection_pipeline

As previously mentioned, it is comprehensive of RGB images of strawberry cluster, and collected trajectories both in task and joint space, saved as numpy arrays.

What you need from the dataset, you can already find it in the description:

        - strawberrry_renamed: 950 RGB images of strawberry cluster 
        - probabilistic_renamed: 95 json files, associated to the strawberry images, containing the probabilistic weights for the ProMPs

### Code 

Respectively in 'e00_autoencoder_f', 'e01_rtp_cnn', 'e01_rtp_cnn_joints', 'preprocessing', you will find all the necessary code to replicate and modify the project.

## Try Yourself!

To reproduce the project, create a folder and name it as you wish. Inside that folder, download the e00, e01s and preprocessing folders. On another folder called 'dataset', download the dataset folders: 'probabilistic_renamed' and 'strawberry_renamed'.

The training phase is done in two steps:

    - Training of the autoencoder: open your terminal, cd to the autoencoder folder, and type "python -m experiment run"
    - Training of the MLP: open your terminal, cd to the e01 folder, and type "python train_test.py"
  
Have fun!

If you have any issues, feel free to contact at: fracastelli98@gmail.com 
 




