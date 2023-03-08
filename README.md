# DeepProMPs_strawberry_rtp

## About The Project 

The aim of the project is to apply the deep probabilistic movement primitives framework in the Agri-robotics filed, to solve the reach to pick task for a strawberry harvest application. 

The action of approaching the strawberry is performed by a Franka Emika robotic arm, equipped with an eye-on-hand RealSense camera. The camera takes an RGB image of the strawberry cluster, which is going to be the input of our deep learning model. 

The architecture of the model is composed of a CNN-Autoencoder, from which we retrieve the latente space of the strawberry cluster image, and a MultiLayerPerceptron, from which we predict the trajectories for the robotic arm to execute. The latent space is used as input for the MLP, that is made of three convolutional layers and two dense layer. 

We trained the MLP model on ProMPs weights, retrieved from a set of trajectories collected by human-expertise. We used a probabilistic approach, defining a trajectory probability distribution of weights by its mean and covariance. This operations where performed both in task and joint space.

You can see a picture of the proposed architecture below:

![working_framwork](https://user-images.githubusercontent.com/82958449/223677540-cab77153-1754-40e3-abbb-bac5c0ffd736.jpg)

From the predicted distribution, sample a trajectory and feed it to the robot.

## What's inside 

### Working environment

Models were built and trained using a conda environment with Python 3.9 and Tensorflow 2.7. You can replicate the same virtual environment from the "environment.yml" with the following command:

        conda env create -f environment.yml -n "NAME_OF_THE_ENV"

### Dataset

The complete raw dataset can be found at this link, where you will also find additional useful informations: https://github.com/imanlab/Franka_datacollection_pipeline

As previously mentioned, it is comprehensive of RGB images of strawberry cluster, and collected trajectories both in task and joint space, saved as numpy arrays.

Download the strawberry images and save them inside a folder. Use 'rename_dataset' in the preprocessing folder to rename the images the right way. Save the renamed images in a folder named 'strawberry_renamed'. This is the folder you will use for the training of your model.

For the probabilisitc weights, there are two folders in the description:

        - probabilistic_renamed: 95 json files, associated to the strawberry images, containing the probabilistic weights for the ProMPs in task space
        - probabilistic_renamed_joints: 95 json files, associated to the strawberry images, containing the probabilistic weights for the ProMPs in joint space 
        
Train your MLP model with these two, respectively.

### Code 

Respectively in 'e00_autoencoder_f', 'e01_rtp_cnn', 'e01_rtp_cnn_joints', 'preprocessing', you will find all the necessary code to replicate and modify the project.

## Try Yourself!

To reproduce the project, create a folder and name it as you wish. Inside that folder, download the e00, e01s and preprocessing folders. On another folder called 'dataset', download the dataset folders, and move strawberry_renamed inside.

The training phase is done in two steps:

    - Training of the autoencoder: open your terminal, cd to the autoencoder folder, and type "python -m experiment run". The autoencoder need to be             trained just on the strawberry images.
    - Training of the MLP: open your terminal, cd to the e01 folder, and type "python train_test.py". Remember to train one model on task space weights and       another model on joint space weights.
  
Have fun!

If you have any issues, feel free to contact at: fracastelli98@gmail.com 
 




