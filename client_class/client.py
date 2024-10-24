import os
import sys
import importlib

sys.path.append("..")
from dataset_creation.dataset_main import create_dataset
from models.model_helpers import compile_model, get_class_weights


class Client:
    def __init__(self, name, config, run_num):
        self.name = name
        self.config = config
        self.results_folder = self.setup_folder(run_num)

        self.train_data = None
        self.train_class_distribution = None
        self.valid_data = None
        self.valid_class_distribution = None
    
        self.model = None
        self.class_weights = None

        self.train_steps_per_epoch = None
        self.valid_steps_per_epoch = None

        self.train_metrics = None
        self.valid_metrics = None
    
        self.callbacks = None


    def setup_folder(self, run_num):
        '''
        check if folder for storing exists and otherwise create it
        '''
        if not os.path.isdir(self.config['model_save_path']):
            os.makedirs(self.config['model_save_path'])
        path = os.path.join(self.config['model_save_path'], run_num)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
        
    
    def create_data(self, data_generation_config, is_federated):
        self.train_data, self.train_class_distribution = create_dataset(data_generation_config,
                                                              client_config=self.config,
                                                              usage_mode='train',
                                                              is_federated=is_federated,
                                                              client_name=self.name
                                                            )

        self.valid_data, self.valid_class_distribution = create_dataset(data_generation_config,
                                                              client_config=self.config,
                                                              usage_mode='valid',
                                                              is_federated=is_federated,
                                                              client_name=self.name
                                                            )

        print(f'number of records in training data set: {sum(self.train_class_distribution)}')
        print(f'number of records in validation dataset: {str(sum(self.valid_class_distribution))}')

        return
    

    def initialize_model(self, model_config, train_config, folder, _log):
        mod = importlib.import_module('models.' + model_config['name'])
        model = getattr(mod, model_config['name'])(model_config)
        
        class_weights, metric_class_weights = get_class_weights(train_config['class_weight'],
                                                                train_config['weighted_metrics'],
                                                                self.train_class_distribution,
                                                                _log)
        self.model = compile_model(model, train_config, metric_class_weights, _log)
        with open(os.path.join(self.results_folder, 'model_json.json'), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.class_weights = class_weights

        return