"""
SCRIPT USE TO RENAME MY DATASET TO ALESSANDRA'S A-LIKE
"""
import json
import os

from skimage.io import imread, imshow, show, imsave
from natsort import natsorted

img_dir = "/home/francesco/PycharmProjects/dataset/dataset_autoencoder/strawberry_bb/"
saving_imgs_dir = "/home/francesco/PycharmProjects/dataset/dataset_autoencoder/strawberry_renamed/"
json_dir = "/home/francesco/PycharmProjects/dataset/dataset_pca/probabilistic_7dof_joints/"
saving_jsons_dir = "/home/francesco/PycharmProjects/dataset/dataset_autoencoder/probabilistic_joints_renamed/"


def load_image(img_path):
    """
    Loads the original RGB image
    """
    img = imread(img_path, pilmode='RGB')
    return img.astype('uint8')


def get_config(filename):
    # config = filename.split(sep='/')[-1].strip('.png').split(sep="_")[3]
    num_config = int(filename.split(sep='/')[-1].strip('.png').split(sep="_")[3].strip("config"))
    num_config = num_config + 1
    renamed_config = "conf" + str(num_config)

    return renamed_config


def get_berry(filename):
    # strawberry = filename.split(sep='/')[-1].strip('.png').split(sep='_')[-2]
    num_strawberry = int(filename.split(sep='/')[-1].strip('.png').split(sep='_')[-2].strip("strawberry"))
    num_strawberry = num_strawberry  + 1
    renamed_berry = "berry" + str(num_strawberry)

    return renamed_berry


def get_traj(filename):
    # traj = filename.split(sep='/')[-1].strip('.png').split(sep='_')[-1]
    num_traj = filename.split(sep='/')[-1].strip('.png').split(sep='_')[-1].strip("traj")
    renamed_traj = "t_00" + num_traj

    return renamed_traj


if __name__ == "__main__":
    """
    img_list_sorted = natsorted(os.listdir(img_dir))
    count = 0
    prev_count = 0
    for filename in img_list_sorted:
        img = load_image(img_dir + filename)
        # imshow(img.astype("uint8"))
        # show()
        if filename.split(sep=".")[-1] == 'png':
            new_count = int(filename.split(sep='/')[-1].strip('.png').split(sep='_')[-2].strip("strawberry"))
            if new_count != prev_count:
                count = count + 1
            config = get_config(filename)
            berry = get_berry(filename)
            traj = get_traj(filename)
            new_name = str(count) + "_" + config + "_" + traj + "_" + berry + ".png"
            # Save the renamed image
            imsave(saving_imgs_dir + new_name, img)
            # Save previous count to check the counter
            prev_count = int(filename.split(sep='/')[-1].strip('.png').split(sep='_')[-2].strip("strawberry"))
    """
    """
    RENAME THE .JSON FILES
    """
    json_list_sorted = natsorted(os.listdir(json_dir))
    i = 0
    for filename in json_list_sorted:
        f = open(json_dir + filename)
        data = json.load(f)
        f.close()
        new_name = str(i) + ".json"
        save_dir = saving_jsons_dir + new_name
        with open(save_dir, "w") as s:
            json.dump(data, s)
        i = i + 1
