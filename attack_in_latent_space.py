import sys
from src.datasets.load_ascadr import *
from src.datasets.load_dpav42 import *
from src.preprocess.generate_hiding_coutermeasures import *
from src.neural_networks.models import *
from src.metrics.guessing_entropy import *
from src.metrics.perceived_information import *
from src.hyperparameters.random_search_ranges import *
from src.datasets.paths import *
from src.training.train import *
from src.utils.utils import *
import gc
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append("/home/nfs/mkrcek/AutoEncodersDLSCA")

if __name__ == "__main__":

    dataset_name = sys.argv[1]
    model_type = sys.argv[2]
    leakage_model = sys.argv[3]
    ae_model = sys.argv[4]
    runs = int(sys.argv[5])
    home_folder = "."
    # home_folder = "/tudelft.net/staff-bulk/ewi/insy/CYS/mkrcek"
    trace_folder = f"{home_folder}/datasets"
    ae_model_file = f"{home_folder}/{ae_model}"

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
    folder_results = f"{home_folder}/{model_type}_{ae_model_name}_{nb_features}"
    scaler = StandardScaler()
    latent_x_profiling = scaler.fit_transform(latent_x_profiling)
    latent_x_attack = scaler.transform(latent_x_attack)

    """ Run random search """
    for search_index in range(runs):
        """ generate hyperparameters """
        hp_values = hp_list(model_type)
        hp_values["seed"] = np.random.randint(1048576)
        print(hp_values)

        """ Create model """
        baseline_model = cnn(dataset.classes, nb_features, hp_values) if model_type == "cnn" else mlp(dataset.classes,
                                                                                                     nb_features,
                                                                                                     hp_values)
        """ Train model """
        model, history = train_ds_diff(baseline_model, latent_x_profiling, dataset.y_profiling,
                                       dataset_parameters['epochs'], hp_values["batch_size"], nb_test_samples)

        """ Compute guessing entropy and perceived information """
        predictions = model.predict(latent_x_attack)
        GE, NT = guessing_entropy(predictions, labels_key_guess, dataset.correct_key, dataset_parameters["n_attack_ge"])
        PI = information(predictions, dataset.attack_labels, dataset.classes)

        """ Save results """
        new_filename = get_filename(folder_results, dataset_name, model_type, leakage_model,)
        hp_values['optimizer'] = hp_values['optimizer'].__name__
        np.savez(new_filename,
                 GE=GE,
                 NT=NT,
                 PI=PI,
                 hp_values=hp_values,
                 history=history.history,
                 dataset=dataset_parameters,
                 encoder_file=ae_model
                 )

        del model
        gc.collect()
