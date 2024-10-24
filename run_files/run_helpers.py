import re
import os
import yaml


def check_config(config):
    # GENERAL
    check_config_general(config)

    # DATA GENERATION

    # the value to monitor (decide which model is best), needs to be one of the metrics or loss
    if config['training']['compile_metrics'] is not None:
        assert config['training']['monitor_val'][4:] in (
            *[re.sub(r'(?<!^)(?=[A-Z])', '_', m).lower() for m in config['training']['compile_metrics']],
            *['loss'], *config['training']['loss_fn']), \
            'monitor_value is not in watchlist'
    else:
        assert config['training']['monitor_val'][4:] in (
            *['loss'], *config['training']['loss_fn']), \
            'monitor_value is not in watchlist'

    assert 'ModelCheckpoint' not in [m['name'] for m in config['training']['callbacks']], \
        'model is automatically saved, please do not use ModelCheckpoint'


def check_config_general(config):
    if config['general']['label_type'] == 'isup':
        assert ('CategoricalAccuracy' not in config['training']['compile_metrics']), \
            'use tf_categorical_accuracy instead of CategoricalAccuracy or categorical_accuracy'
        assert ('tf_f1_score' not in config['training']['compile_metrics']), 'f1 not valid for fillin encoding'
        assert (config['model']['name'] == 'm_isup'), 'For ISUP classification, please use model m_isup'

    assert config['general']['number_of_classes'] is not None, 'Please specify number of classes in config.general'
    assert isinstance(config['general']['number_of_classes'], int), \
        'Please specify number_of_classes in config.general as int'
    assert config['general']['number_of_classes'] > 0, \
        'Please specify number_of_classes in config.general greater than 0'

    # data directory should not be empty
    for client_name in config['general']['client_names']:
        client_config_file = os.path.join(os.getcwd(), '..', config['general']['folder'], client_name, 'client_config.yaml')
        with open(client_config_file, 'r') as file:
            client_config = yaml.safe_load(file)
    
        dir = client_config['image_directory']
        assert os.listdir(dir) != [], "Image directory {} is empty".format(dir)

    assert (config['general']['image_channels'] == 3), 'currently, only RGB implemented, so set img_channels to 3'


def check_data_generation_config(config):
    for client_name in config['general']['client_names']:
        client_config_file = os.path.join(os.getcwd(), '..', config['general']['folder'], client_name, 'client_config.yaml')
        with open(client_config_file, 'r') as file:
            client_config = yaml.safe_load(file)
        assert os.path.isfile(client_config['train_csv_file']), 'train csv file not found'
        assert os.path.isfile(client_config['valid_csv_file']), 'valid csv file not found'
        assert os.path.isfile(client_config['test_csv_file']), 'test csv file not found'

    assert config['data_generation']['train_batch_size'] is not None, 'please specify a training batch size'
    assert config['data_generation']['valid_batch_size'] is not None, 'please specify a validation batch size'


def check_model_config(config):
    assert config['model'] in ['m_isup'], 'Please choose as model m_isup'
    assert isinstance(config['model']['dense_layer_nodes'], list), 'Provide number of dense layers as list'


def check_training_config(config):
    assert config['epochs'] is not None, 'Must provide number of epochs as int'
    assert isinstance(config['epochs'], int), 'Must provide number of epochs as int'
