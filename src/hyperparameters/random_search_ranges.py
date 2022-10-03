import itertools
import numpy as np
from tensorflow.keras.optimizers import *


def hp_list(model_type, latent_dim=None):
    if model_type == "mlp":
        hp = {
            "layers": [1, 2, 3, 4, 5, 6],
            "neurons": [20, 40, 50, 100, 150, 200, 300, 400],
            "batch_size": [100, 200, 400],
            "activation": ["elu", "selu", "relu"],
            "learning_rate": [0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad],  # , Adamax, Adadelta]
        }
    elif model_type == "cnn":
        hp = {
            "neurons": [20, 50, 100, 200],
            "batch_size": [100, 200, 400],
            "layers": [1, 2],
            "filters": [4, 8, 12, 16],
            "kernel_size": [10, 20, 30, 40],
            "strides": [5, 10, 15, 20],
            "pool_size": [2],
            "pool_strides": [2],
            "conv_layers": [1, 2, 3, 4, 5],
            "activation": ["elu", "selu", "relu"],
            "learning_rate": [0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad],
        }
    elif model_type == "ae_mlp":
        hp = {
            "layers": [1, 2, 3, 4, 5, 6],
            "neurons": [20, 50, 100, 250],
            "batch_size": [100, 200, 400],
            "activation": ["tanh", "elu", "selu", "sigmoid"],
            "learning_rate": [0.005, 0.001, 0.0001, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad]
        }
    elif model_type == "ae_cnn":
        hp = {
            "batch_size": [100, 200, 400],
            "filters": [4, 8, 16],
            "kernel_size": [10, 20],  # , 30, 40],
            "strides": [5, 10],  # , 15, 20],
            "pool_size": [2, 4],
            "pool_strides": [2, 4],
            "pooling_type": ["Avg", "Max"],
            "conv_layers": [1, 2, 3, 4],
            "activation": ["tanh", "elu", "selu", "sigmoid"],
            "learning_rate": [0.005, 0.001, 0.0001, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad]
        }
    elif model_type == "ae_mlp_dcr":
        nb_layers = np.random.choice([1, 2, 3, 4, 5, 6])
        neurons = np.array([20, 40, 50, 100, 150, 200, 300, 400])
        architecture = sorted(np.random.choice(neurons, nb_layers))

        hp = {
            "architecture": [architecture],
            "batch_size": [100, 200, 400],
            "activation": ["tanh", "elu", "selu", "sigmoid"],
            "learning_rate": [0.005, 0.001, 0.0001, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad],
        }
    elif model_type == "ae_mlp_str_dcr":  # strictly decreasing
        nb_layers = np.random.choice([1, 2, 3, 4, 5, 6])
        neurons = np.array([20, 40, 50, 100, 150, 200, 300, 400])
        if latent_dim is None:
            raise ValueError(f"For model type {model_type}, latent_dim has to be set.")
        if latent_dim > np.max(neurons):
            raise ValueError(f"Latent dim has to be larger {np.max(neurons)}.")
        neurons = neurons[np.where(neurons >= latent_dim)]
        architecture = sorted(np.random.choice(neurons, nb_layers))

        hp = {
            "architecture": [architecture],
            "batch_size": [100, 200, 400],
            "activation": ["tanh", "elu", "selu", "sigmoid"],
            "learning_rate": [0.005, 0.001, 0.0001, 0.00001],
            "weight_init": ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal",
                            "glorot_normal"],
            "optimizer": [Adam, RMSprop, SGD, Adagrad],
        }
    else:
        raise ValueError(f"wrong model type {model_type}")

    keys, value = zip(*hp.items())
    search_hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*value)]
    print(f"Total search space for {model_type}: {len(search_hp_combinations)}")

    hp_id = np.random.randint(len(search_hp_combinations))
    return search_hp_combinations[hp_id]
