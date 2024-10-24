import os
import sys
import json
import yaml

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

sys.path.append("..")
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from client_class.client import Client
from settings.sacred_experiment import ex
from run_files.run_helpers import check_config
from training_evaluation.train import train_loop
from training_evaluation.evaluate import evaluate
from dataset_creation.dataset_main import create_test_dataset


def print_system_summary(_log):
    # system summary
    _log.debug('PATH: {}'.format(sys.path))
    _log.debug('current working directory: {}'.format(os.getcwd()))
    _log.info('Num GPUs Available: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    _log.debug('devices:{} '.format(device_lib.list_local_devices()))
    _log.info('tensorflow version {}'.format(tf.__version__))


def run(_config, _log, _run):
    """
    this is the main function that runs training and test of the model once
    Returns: best result, based on monitoring values in config
    """
    # define threads used
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    print_system_summary(_log)

    # checks on the config
    check_config(_config)
    np.random.seed(_config['data_generation']['seed'])
    tf.random.set_seed(_config['data_generation']['seed'])

    clients = list()
    # DATA GENERATION TRAINING AND VALIDATION
    for client_name in _config['general']['client_names']:
        client_config_file = os.path.join(os.getcwd(), '..', _config['general']['folder'], client_name, 'client_config.yaml')
        with open(client_config_file, 'r') as file:
            client_config = yaml.safe_load(file)
        client = Client(client_name, config=client_config, run_num=str(_run._id))

        if len(_config["general"]["client_names"]) == 1:
            is_federated = False
        else:
            is_federated = True
        client.create_data(data_generation_config=_config['data_generation'], is_federated=is_federated)
        client.initialize_model(_config['model'], _config['training'], _config['general']['folder'], _log)
        clients.append(client)

    
    # set same model on each client
    server_weights = clients[0].model.get_weights()
    for client in clients:
        client.model.set_weights(server_weights)
    
    
    # MODEL TRAINING
    best_weights, history = train_loop(clients, 
                                _config['data_generation']['train_batch_size'],
                                _config['data_generation']['valid_batch_size'],
                                _config['training'],
                                _config['data_generation']['label_type'],
                                _run,
                                _log=None)

    for client in clients:
        client.model.set_weights(best_weights)


    # MODEL EVALUATION (only on coordinator)
    server = clients[0]
    test_results = list()
    if isinstance(server.config["test_csv_file"], str):
        server.config["test_csv_file"] = [server.config["test_csv_file"]]

    for filename in server.config["test_csv_file"]:
        if filename.split("/")[-1].split(".")[0][:4] in ["TMA_", "spl_"]:
            i = int(filename.split("/")[-1].split(".")[0][4:])
        else:
            i = 1
        print(i)
        test_data, test_class_distribution = create_test_dataset(data_generation_config=_config['data_generation'],
                                                            client_config=server.config,
                                                            filename=filename,
                                                            usage_mode='test',
                                                            test_id=i
                                                            )

        if len(server.config["test_csv_file"]) != 1:
            res_path = os.path.join(client.results_folder, str(i))
        else:
            res_path = server.results_folder
        if not os.path.exists(res_path):
            os.mkdir(res_path)

        test_results.append(evaluate(server.model, test_data, _config['data_generation']['label_type'],
                                test_class_distribution, _config['evaluation']['metrics'], res_path))

    # STORE RESULTS
    train_history = {}
    for k, v in history.history.items():
        train_history[k] = np.array(v).astype(float).tolist()
    allresults = {'train_results': train_history, 'test_results': test_results}

    for client in clients:
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', client.results_folder))
        with open(os.path.join(script_path, 'temp_results.json'), 'w') as resultsfile:
            json.dump(allresults, resultsfile)
        ex.add_artifact(os.path.join(script_path, 'temp_results.json'), f'{client.name}results.json')
        os.remove(os.path.join(script_path, 'temp_results.json'))


@ex.automain
def main(_config, _log, _run):
    run(_config, _log, _run)


