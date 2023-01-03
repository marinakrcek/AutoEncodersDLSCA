import sys
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_dpav42 import *
from src.preprocess.generate_hiding_coutermeasures import *
from src.neural_networks.models import *
from src.metrics.guessing_entropy import *
from src.metrics.perceived_information import *
from src.hyperparameters.random_search_ranges import *
from tensorflow.keras.optimizers import *
from src.datasets.paths import *
from src.training.train import *
from src.utils.utils import *
import gc
import os
import pandas as pd


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append("/home/nfs/mkrcek/AutoEncodersDLSCA")

if __name__ == "__main__":

    dataset_name = sys.argv[1]
    model_type = sys.argv[2]
    leakage_model = sys.argv[3]
    ae_model = sys.argv[4]
    attack_model = sys.argv[5]
    max_nb_epochs = int(sys.argv[6])
    home_folder = "."
    # home_folder = "/tudelft.net/staff-bulk/ewi/insy/CYS/mkrcek"
    trace_folder = f"{home_folder}/datasets"
    ae_model_file = f"{home_folder}/{ae_model}"
    attack_model_file = f"{home_folder}/{attack_model}"

    dataset_parameters = None
    class_name = None

    if dataset_name == "dpa_v42":
        dataset_parameters = {
            "n_profiling": 70000,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 12,
            "npoi": 2000,
            "epochs": 100
        }
        class_name = ReadDPAV42
    if dataset_name == "ascad-variable":
        dataset_parameters = {
            "n_profiling": 200000,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 2,
            "npoi": 1400,
            "epochs": 100
        }
        class_name = ReadASCADr
    if dataset_name == "ascadf":
        dataset_parameters = {
            "n_profiling": 50000,
            "n_attack": 5000,
            "n_attack_ge": 3000,
            "target_byte": 2,
            "npoi": 700,
            "epochs": 100
        }
        class_name = ReadASCADf

    """ Create dataset """
    dataset = class_name(
        dataset_parameters["n_profiling"],
        dataset_parameters["n_attack"],
        file_path=get_dataset_filepath(trace_folder, dataset_name, dataset_parameters["npoi"]),
        target_byte=dataset_parameters["target_byte"],
        leakage_model=leakage_model,
        first_sample=0,
        number_of_samples=dataset_parameters["npoi"]
    )

    """ Rescale and reshape (if CNN) """
    dataset.rescale(True if model_type.endswith("cnn") else False)

    """ Generate key guessing table """
    labels_key_guess = dataset.labels_key_hypothesis_attack

    """ load encoder to create dataset in latent space """

    encoder = load_model(ae_model_file)
    # get the dataset in latent space
    latent_x_profiling = encoder.predict(dataset.x_profiling)
    nb_train_samples, nb_features = latent_x_profiling.shape
    latent_x_attack = encoder.predict(dataset.x_attack)
    nb_test_samples = latent_x_attack.shape[0]
    ae_model_name = ae_model[:ae_model.find(f"_{nb_features}")]
    folder_results = f"{home_folder}/best_ascadf/TL/with_{ae_model_name}_{nb_features}"
    scaler = StandardScaler()
    latent_x_profiling = scaler.fit_transform(latent_x_profiling)
    latent_x_attack = scaler.transform(latent_x_attack)

    """ load hyperparameters """
    attack_model = load_model(attack_model_file)
    optimizers = {"Adam": Adam, 'RMSprop': RMSprop, 'SGD': SGD, 'Adagrad': Adagrad}
    npz_file = np.load(attack_model_file+".npz", allow_pickle=True)
    hp_values = npz_file['hp_values'].item()
    hp_values['optimizer'] = optimizers[hp_values['optimizer']]
    print(hp_values)

    GEs = []
    NTs = []
    PIs = []
    print('nb epochs\tmin GE\tNT')
    for nb_epochs in range(max_nb_epochs):
        attack_model.trainable = False
        attack_model.layers[-1].trainable = True  ## freeze all except last output layer
        # pd.set_option('max_colwidth', -1)
        # layers = [(layer, layer.name, layer.trainable) for layer in attack_model.layers]
        # pdfr = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

        """ Train model """
        attack_model, history = train_ds_diff(attack_model, latent_x_profiling, dataset.y_profiling,
                                              0 if nb_epochs == 0 else 1, hp_values["batch_size"], nb_test_samples)

        """ Compute guessing entropy and perceived information """
        predictions = attack_model.predict(latent_x_attack)
        GE, NT = guessing_entropy(predictions, labels_key_guess, dataset.correct_key, dataset_parameters["n_attack_ge"])
        PI = information(predictions, dataset.attack_labels, dataset.classes)
        minGE = np.min(GE)
        print(f"{str(nb_epochs)}\t{str(minGE)}\t{str(NT)}")
        GEs.append(GE)
        NTs.append(NT)
        PIs.append(PI)
        if minGE <= 1:
            break
    """ Save results """
    new_filename = get_filename(folder_results, dataset_name, model_type, leakage_model,)
    hp_values['optimizer'] = hp_values['optimizer'].__name__
    np.savez(new_filename,
             GEs=GEs,
             NTs=NTs,
             PIs=PIs,
             hp_values=hp_values,
             dataset=dataset_parameters,
             encoder_file=ae_model,
             attack_model=attack_model_file,
             nb_epochs=nb_epochs,
             max_nb_epochs=max_nb_epochs
             )
    del attack_model
    gc.collect()