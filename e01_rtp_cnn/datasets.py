"""
This module contains the dataset directly usable in the experiments.
Those in the data module are just base classes, not directly instantiable.
"""
import tensorflow_probability as tfp
import data

tfd = tfp.distributions


class Dataset_RGBProMP(data.DatasetOrdinary, data.DatasetRGB):

    # super() e' un inheritance => permette di riprendere le caratteristiche di DatasetOrdinary e di DatasetRGB
    def __init__(self, dataset_dir, rgb_dir, encoder, **kwargs):
        super().__init__(dataset_dir=dataset_dir, rgb_dir=rgb_dir, encoder=encoder, **kwargs)

    def prepare_data(self, val_frac=0.0, N_test=0, random_state=None, use_val_in_train=False):
        idx_train, idx_val, idx_test = self._split_train_test_val()
        n_train = len(idx_train)
        n_test = len(idx_test)
        n_val = len(idx_val)
        # Create the datasets splitting in train test and val, and coupling the encoded image with its respective weights
        X_train, y_train = {"encoded": self.samples["img_enc"][idx_train].astype('float64')}, \
            {"mean_weights": self.samples["mean_weights"][idx_train].reshape((n_train, 56)).astype('float64'),
             "cov_weights": self.samples["cov_weights"][idx_train].reshape((n_train, 63)).astype('float64')}

        X_val, y_val = {"encoded": self.samples["img_enc"][idx_val].astype('float64')}, \
            {"mean_weights": self.samples["mean_weights"][idx_val].reshape((n_val, 56)).astype('float64'),
             "cov_weights": self.samples["cov_weights"][idx_val].reshape((n_val, 63)).astype('float64')}

        X_test, y_test = {"encoded": self.samples["img_enc"][idx_test].astype('float64')}, \
            {"mean_weights": self.samples["mean_weights"][idx_test].reshape((n_test, 56)).astype('float64'),
             "cov_weights": self.samples["cov_weights"][idx_test].reshape((n_test, 63)).astype('float64')}

        self.data = (X_train, y_train), (X_val, y_val), (X_test, y_test)
        self.data_names = {'train_ids': [self.samples["images_id"][x] for x in idx_train],
                           'val_ids': [self.samples["images_id"][x] for x in idx_val],
                           'test_ids': [self.samples["images_id"][x] for x in idx_test]}
