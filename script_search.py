import sys
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('GPUs:', gpus)
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    # if tf.test.gpu_device_name():
    #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    dataset_name = sys.argv[1]
    model_type = sys.argv[2]
    latent_dim = int(sys.argv[3])
    runs = int(sys.argv[4])
    leakage_model = 'HW'
    # trace_folder = "./datasets"
    trace_folder = "/home/nfs/mkrcek/datasets"
    # folder_results = f"./{model_type}/"
    folder_results = f"/home/nfs/mkrcek/{model_type}"

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

    """ Run random search """
    for search_index in range(runs):
        """ generate hyperparameters """
        hp_values = hp_list(model_type, latent_dim=latent_dim)
        hp_values["seed"] = np.random.randint(1048576)
        hp_values['latent_dim'] = latent_dim
        print(hp_values)

        """ Create model """
        if model_type == "ae_mlp":
            encoder, decoder, autoencoder = autoencoder_mlp(hp_values['latent_dim'], dataset.ns, hp_values)
        elif model_type == "ae_cnn":
            encoder, decoder, autoencoder = autoencoder_cnn(hp_values['latent_dim'], dataset.ns, hp_values)
        elif model_type == "ae_mlp_dcr" or model_type == "ae_mlp_str_dcr":
            encoder, decoder, autoencoder = autoencoder_mlp_dcr(hp_values['latent_dim'], dataset.ns, hp_values)
        else:
            raise ValueError(f"wrong model type {model_type}")
        """ Train model """
        model, history = train_ae_model(autoencoder, dataset, dataset_parameters['epochs'], hp_values["batch_size"])

        """ Get some predictions for MSE """
        from sklearn.metrics import mean_squared_error

        predictions = model.predict(dataset.x_attack)
        if model_type.endswith('cnn'):
            dataset.x_attack = np.reshape(dataset.x_attack, (dataset.x_attack.shape[0], dataset.x_attack.shape[1]))
        mse_ds = mean_squared_error(dataset.x_attack, predictions)
        # print(mse_ds)
        mse = np.square(dataset.x_attack - predictions)

        # because dataset is with HW leakage model
        possibilities = 9
        snr_orig = CalculateSNR(dataset.x_attack, dataset.share2_attack, possibilities)
        snr_pred = CalculateSNR(predictions, dataset.share2_attack, possibilities)
        encoded_data = np.array(encoder(dataset.x_attack))
        snr_encd = CalculateSNR(encoded_data, dataset.share2_attack, possibilities)

        mask_possibilities = len(np.unique(dataset.share1_attack))
        snr_orig_mask = CalculateSNR(dataset.x_attack, dataset.share1_attack, mask_possibilities)
        snr_pred_mask = CalculateSNR(predictions, dataset.share1_attack, mask_possibilities)
        snr_encd_mask = CalculateSNR(encoded_data, dataset.share1_attack, mask_possibilities)
        snr_HW = {'snr_orig': snr_orig, 'snr_pred': snr_pred, 'snr_encd': snr_encd, 'snr_orig_mask': snr_orig_mask,
                  'snr_pred_mask': snr_pred_mask, 'snr_encd_mask': snr_encd_mask}

        datasetID = class_name(
            dataset_parameters["n_profiling"],
            dataset_parameters["n_attack"],
            file_path=get_dataset_filepath(trace_folder, dataset_name, dataset_parameters["npoi"]),
            target_byte=dataset_parameters["target_byte"],
            leakage_model='ID',
            first_sample=0,
            number_of_samples=dataset_parameters["npoi"]
        )
        datasetID.rescale(False)
        # same for when dataset is with leakage model ID
        possibilities = 256
        snr_orig = CalculateSNR(datasetID.x_attack, datasetID.share2_attack, possibilities)
        snr_pred = CalculateSNR(predictions, datasetID.share2_attack, possibilities)
        encoded_data = np.array(encoder(datasetID.x_attack))
        snr_encd = CalculateSNR(encoded_data, datasetID.share2_attack, possibilities)
        snr_ID = {'snr_orig': snr_orig, 'snr_pred': snr_pred, 'snr_encd': snr_encd}

        smse = np.sum(mse, axis=1)
        lowest_mse_i = np.argmin(smse)
        orig_data = dataset.x_attack[lowest_mse_i]
        pred = predictions[lowest_mse_i]

        """ Save results """
        new_filename = get_filename(folder_results, dataset_name, model_type, leakage_model, )
        hp_values['optimizer'] = hp_values['optimizer'].__name__
        np.savez(new_filename,
                 mse_ds=mse_ds,
                 mse=mse,
                 best_orig=orig_data,
                 best_pred=pred,
                 hp_values=hp_values,
                 history=history.history,
                 dataset=dataset_parameters,
                 snr_HW=snr_HW,
                 snr_ID=snr_ID
                 )
        encoder.save(new_filename[:-4])
        del model
        gc.collect()
