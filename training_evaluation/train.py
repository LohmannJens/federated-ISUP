import os
import re
import time
import logging

import numpy as np
import tensorflow as tf

from settings.sacred_experiment import ex
from tensorflow.keras.callbacks import Callback

from dataset_creation.label_encoding import label_to_int
from aggregation.fed_avg import fed_avg


def value_name_in_logs(logs, value):
    """
        For each client a number is added to the end of the value names.
        This is for example for metrics and monitor values the case.
        This function checks which number was added for the corresponding client.
        The function returns the new name of the value.
        This works for up to 20 clients.
    """
    for i in range(21):
        if i == 0:
            new_val = value
        else:
            new_val =  f"{value}_{i}"
        if logs.get(new_val) or logs.get(new_val) == 0.0:
            return new_val
    print(f"value {value} does not exist")
    return

class LogPerformance(Callback):
    """
    Self defined Callback function, which at the moment calls an internal function at the end of each training epoch to
    log the metrics, save each model and delete old model if there was a better one afterwards
    """
    def __init__(self, train_params, run_id, model_path):
        """
        initialization
        :param train_params: dictionary with 'initial_epoch': int
                                             'monitor_val': string (which metric/loss to use to decide for best model)
                                             'model_save_path': string, where to save model
        :param run_id:
        """
        self.train_params = train_params
        self.prev_metric_value = -9999.9
        self.curr_epoch = int(train_params['initial_epoch'])
        self.best_epoch = self.curr_epoch
        self.monitor = train_params['monitor_val']
        self.model_path = model_path

        super().__init__()
    @ex.capture
    def on_epoch_end(self, epoch, logs, _run):
        """
        at end of each epoch, it should be evaluated whether the model is better now (based on monitor_val) and should
        be saved or not
        :param data: tupel with:
                        current epoch, needed to save model
                        path where to save the model
        :param logs: current metrics and losses
        :return:
        """
        self.log_performance(train_params=self.train_params, logs=logs, _run=_run)


    def better_than_best_epoch(self, logs, monitor):
        """
        compare the current model to the previous ones
        for deleting the weights which do not belong to the best model/epoch
        own function b/c accuracy needs to be higher to be better, but loss needs to be lower

        :param logs: current metric/loss
        :return: True or False whether current model is best
        """
        if ('accuracy' in monitor) or ('f1' in monitor) or ('kappa' in monitor):
            if logs.get(monitor) > self.prev_metric_value:
                return True
            else:
                return False
        elif 'loss' in monitor:
            if logs.get(monitor) < abs(self.prev_metric_value):
                return True
            else:
                return False

    @ex.capture
    def log_performance(self, train_params, _run, logs):
        """
        this logs the loss and all defined metrics from config.yaml, so they are saved and plotted in the mongoDB

        :param _run: parameter from sacred experiment
        :param logs: keras Callback logs

        Returns:

        """
        # Changed: need to loop, because the metrics have different names for different clients
        if train_params["compile_metrics"] is not None:
            for metric in train_params["compile_metrics"]:
                metrics = re.sub(r'(?<!^)(?=[A-Z])', '_', metric).lower()
                metrics = value_name_in_logs(logs, metrics)
                _run.log_scalar(metrics, float(logs.get(metrics)))
                _run.log_scalar("val_"+metrics, float(logs.get('val_'+metrics)))
        _run.log_scalar("loss", float(logs.get('loss')))
        _run.log_scalar("val_loss", float(logs.get('val_loss')))
            
        self.curr_epoch += 1


@tf.function
def fiveepochlower(epoch, lr):
    """
    halve learning rate every five epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 5 == 0) and epoch != 0:
        lr = lr/2
    return lr


def tenepochlower(epoch, lr):
    """
    halve learning rate every ten epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 10 == 0) and epoch != 0:
        lr = lr/2
    return lr


def twentyepochlower(epoch, lr):
    """
    halve learning rate every ten epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 20 == 0) and epoch != 0:
        lr = lr/2
    return lr


def thirtyepochlower(epoch, lr):
    """
    halve learning rate every ten epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 30 == 0) and epoch != 0:
        lr = lr/2
    return lr

def fourtyepochlower(epoch, lr):
    """
    halve learning rate every ten epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 40 == 0) and epoch != 0:
        lr = lr/2
    return lr


@ex.capture
def list_callbacks(train_params, model_path, _run, _log):
    """
    This function returns a customized callback function (which logs the metrics) and can also add more standard
    keras callbacks

    :param train_params: _config["train"] or a dictionary with info about epochs, callbacks, ...
    :param _run: the _id of _run is important to be able to save results in correct folder

    :return:

    """
    callbacks = []
    callbacks.append(LogPerformance(train_params, _run._id, model_path))
    valid_schedulers = {'fourtyepochlower': fourtyepochlower,
                        'thirtyepochlower': thirtyepochlower,
                        'twentyepochlower': twentyepochlower,
                        'tenepochlower': tenepochlower,
                        'fiveepochlower': fiveepochlower}

    if train_params['callbacks'] is not None:
        for t in train_params['callbacks']:
            if t['name'] == 'LearningRateScheduler':
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(schedule=valid_schedulers[t['params']['schedule']]))
            else:
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(**t['params']))
            _log.debug(type(callbacks[-1]))
            _log.debug(t['params'])
    
    return callbacks

#@tf.function
def training_step(model, datapoint, class_weights, label_type):
    """
    run model on input data, compute the loss and update the weights

    :param model: tensorflow model
    :param datapoint: datapoint as (tf.data) dict with labels and images
    :param class_weights: None or list of how many examples of each class exist, in order to weight samples
    :param label_type: 'isup'
    :return:
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        try:
            int_of_class_weights = tf.cast(label_to_int(label, label_type), 'int32')
        except Exception as error:
            print(error)

        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None

    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = model.loss(y_true=label, y_pred=prediction)

        if sample_weights is not None:
            loss = tf.multiply(loss, sample_weights)
        loss = tf.reduce_mean(loss)    

    gradients = tape.gradient(loss, model.trainable_variables)
    clip_value = 1.0
    #print(max(gradients))
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return prediction, loss


#@tf.function
def valid_step(model, datapoint, class_weights, label_encoding):
    """
    validation step: run model on batch and evaluate loss, no weight update
    :param model: tensorflow/keras model
    :param datapoint: dictionary with
                      'images' tf.Tensor [batchsize, h, w, 3] and
                     'labels' tf.Tensor [batch_size, n_classes (maybe+1 for censoring information)]
    :param class_weights: None or array with weight per class
    :param label_encoding: 'isup'

    :return: prediction tf.Tensor and loss tf.Tensor (scalar value though)
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        int_of_class_weights = tf.cast([label_to_int(label, label_encoding)[i] for i in range(label.shape[0])], 'int32')
        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None
    
    prediction = model(img, training=False)
    loss = model.loss(y_true=label, y_pred=prediction)

    if sample_weights is not None:
        loss = loss * sample_weights
    loss = tf.reduce_mean(loss)
    return prediction, loss


def setup(num_classes, model):
    """
    for training and validation loop, the loss, error, data and prediction need to be (re)set each epoch
    :param num_classes: int, number of classes, needed for confusion matrix in classification
    :param model: tensorflow model, for which the metrics should be reset
    :return:
    """
    loss = 0
    error = 999
    data_storage = {'labels': list(), 'censored': list()}
    prediction_storage = list()
    cm = np.zeros((num_classes, num_classes))
    for metric in model.metrics:
        metric.reset_states()
    for metric in model.compiled_metrics._metrics:
        metric.reset_states()
    return loss, error, data_storage, prediction_storage, cm, model

#@tf.function
def update_standard_metrics(model, data_batch, prediction_batch, batch_size):
    """
    most metrics can be updated here, only some are left out (the ones that require more than a few data points)
    :param model: tensorflow / keras model
    :param data_batch: dictionary with 'images': tf.Tensor and 'labels': tf.Tensor
    :param prediction_batch: tf.Tensor (batch_size, n_classes)
    :param batch_size: TODO: remove b/c can be read from prediction shape?
    :return:
    """
    batch_size = prediction_batch.shape[0]
    for metric in model.metrics:
        label_batch = data_batch['labels']
        if metric.name in ['cohens_kappa'] and batch_size == 1:
            pass  # only calculate in the end
        elif metric.name in ['c_index_censor']:
            pass
        else:
            if 'censor' in metric.name:
                censored_batch = data_batch['censored']
                label_batch = tf.concat((np.array(label_batch, 'float32'), tf.expand_dims(np.array(censored_batch, 'float32'), 1)), 1)
            metric.update_state(label_batch, prediction_batch)

    if len(model.metrics) == 0:
        for metric in model.compiled_metrics._metrics:
            label_batch = data_batch['labels']
            metric.update_state(label_batch, prediction_batch)


@tf.function
def update_other_metrics(model, data_gathered, prediction_gathered):
    """
    update cindex and cohens kappa metrics, since they require more datapoints than just a single batch for calculation
    :param model: tensorflow /keras model
    :param data_gathered:
    :param prediction_gathered:
    :return:
    """
    label_gathered = data_gathered['labels']
    for metric in model.metrics:
        if metric.name in ['cohens_kappa', 'c_index_censor']:
            if 'censor' in metric.name:
                censored_gathered = data_gathered['censored']
                label_gathered = tf.concat((tf.expand_dims(np.array(label_gathered, 'float32'), 1),
                                         tf.expand_dims(np.array(censored_gathered, 'float32'), 1)), 1)
            metric.update_state(label_gathered, prediction_gathered)


def handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                              model, dpt_idx, train_steps_per_epoch, label_type):
    """
    metrics cohens kappa and cindex cannot be calculated on single batches during training, therefore information over
    multiple batches needs to be stored and the evaluation is only done every 10 epochs
    :param pred_storage: former predictions
    :param data_storage: former data
    :param prediction_batch: current prediction
    :param datapoint_batch: current data
    :param model: tensorflow model
    :param dpt_idx: current epoch
    :param train_steps_per_epoch: int, how many steps per epoch (to make sure evaluation is done in last epoch for sure)
    :param label_type: to turn label into integer, needs to know if it is binary, isup_classification or survival
    :return: model (maybe with updated metrics), updated or reset prediction storage and updated or reset data storage
    """
    label_batch = datapoint_batch['labels']
    pred_storage.extend(label_to_int(prediction_batch, label_type))
    data_storage['labels'].extend(label_to_int(label_batch, label_type))
    try:
        data_storage['censored'].extend(datapoint_batch['censored'])
    except:
        pass
    if ((dpt_idx % 10 == 0) and (dpt_idx != 0)) or (dpt_idx >= train_steps_per_epoch):
        update_other_metrics(model, data_gathered=data_storage, prediction_gathered=pred_storage)
        pred_storage = list()
        data_storage = {'labels': list(), 'censored': list()}
    return model, pred_storage, data_storage


@ex.capture
def train_loop(clients, train_batch_size, valid_batch_size, train_params, label_type, _run, _log=None):
    """

    :param model: tensorflow / keras model
    :param train_dataset: tf.data.Dataset with a dictionary structure, so ['images'] and ['labels'] can be accessed
    :param valid_dataset: tf.data.Dataset with a dictionary structure, so ['images'] and ['labels'] can be accessed
    :param train_batch_size: int
    :param valid_batch_size: int
    :param train_params: dictionary with at least
                         epochs: int, global epochs, how many times to aggregate the models
                         initial_epoch: int, 0 if it is a new training, higher if training is resumed
                         local_epochs: int, how many epochs to train per communication round
    :param train_class_distribution: array with size [n_classes,1], for each class the number indicates how many
                                     examples of this class are present in the dataset
    :param valid_class_distribution: array with size [n_classes,1], for each class the number indicates how many
                                     examples of this class are present in the dataset
    :param class_weights: None or array with class weights to be applied to each class (depends on class distribution)
    :param label_type: 'isup' - needed to calculate integer class from array
    :param experiments_dir: path to where the results should be saved (best model, intermediate loss, metrics, ...)
    :param _run: current run (id is needed for saving path) - defined by sacred
    :param _log: to print some intermediate results and the progress
    :return:
    """
    # INITIALIZATION
    if _log is None:
        _log = logging.getLogger('logger')
    initial_epoch = train_params['initial_epoch']
    epochs = train_params['epochs']
    local_epochs = train_params['local_epochs']

    for client in clients:
        client.train_steps_per_epoch = int(sum(client.train_class_distribution) / train_batch_size)
        if client.train_steps_per_epoch == 0:
            client.train_steps_per_epoch = 1
        client.valid_steps_per_epoch = int(sum(client.valid_class_distribution) / valid_batch_size)
        if client.valid_steps_per_epoch == 0:
            client.valid_steps_per_epoch = 1

    for client in clients:
        client.callbacks = list_callbacks(train_params, client.config['model_save_path'], _run=_run, _log=_log)
        # at least, History() should be used as callback
        client.callbacks.append(tf.keras.callbacks.History())
        for callback in client.callbacks:
            callback.set_model(client.model)
            callback.on_train_begin({m.name: m.result() for m in client.model.metrics})
            callback.on_epoch_begin(initial_epoch)


    #######################################
    # loop over epochs
    #######################################

    training_start = time.time()
    best_epoch = 0
    if "loss" in train_params['monitor_val']:
        best_metric = 9999
    else:
        best_metric = 0.0
    for epoch in np.arange(initial_epoch, epochs):
        print(f"### EPOCH {epoch} (communication round) ###")
        s = time.time()
        for client in clients:
            # update callbacks
            for callback in client.callbacks:
                callback.on_epoch_begin(epoch)

        #######################################
        # loop over local epochs
        #######################################
        for local_epoch in range(0, local_epochs):
            print(f"## LOCAL EPOCH {local_epoch} ##")

            for client in clients:
                train_loss, train_error, data_storage, pred_storage, cm, client.model = \
                    setup(client.train_class_distribution.shape[0], client.model)

                #######################################################################################
                # TRAINING
                #######################################################################################
                # iteration over train batches
                for dpt_idx, datapoint_batch in enumerate(client.train_data.take(client.train_steps_per_epoch)):
                    # make prediction and calculate loss for this batch
                    prediction_batch, loss_batch = training_step(client.model, datapoint_batch, client.class_weights, label_type)
                    train_loss = train_loss + np.mean(loss_batch)
                    update_standard_metrics(client.model, datapoint_batch, prediction_batch, train_batch_size)

                    if (len([metric.name for metric in client.model.metrics if metric.name in ['cohens_kappa', 'c_index_censor']])) > 0:
                        handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                                                client.model, dpt_idx, client.train_steps_per_epoch, label_type)

                client.train_metrics = {**{m.name: m.result() for m in client.model.compiled_metrics._metrics}, 'loss': train_loss/(dpt_idx+1)}

        #######################################################################################
        # AGGREGATION
        #######################################################################################
        models = [cl.model for cl in clients]
        if train_params['aggregation_method'] == 'fed_avg':
            n_datapoints = [sum(cl.train_class_distribution) for cl in clients]
            final_weights = fed_avg(models, n_datapoints)
        else:
            exit(f"Aggregation method '{train_params['aggregation_method']}' not implemented.")

        for client in clients:
            client.model.set_weights(final_weights)

        # if optimizer is Nadam also aggregate and update the momentum
        if train_params["optimizer"]["name"] == "Nadam":
            momentum_states = [cl.model.optimizer._momentums for cl in clients]
            velocity_states = [cl.model.optimizer._velocities for cl in clients]
            new_momentum = [tf.reduce_mean([m[j] for m in momentum_states], axis=0) for j in range(len(momentum_states[0]))]
            new_velocity = [tf.reduce_mean([v[j] for v in velocity_states], axis=0) for j in range(len(velocity_states[0]))]
            for client in clients:
                client.model.optimizer._momentums = new_momentum
                client.model.optimizer._velocities = new_velocity

        server = clients[0]
        server.model.save_weights(os.path.join(server.results_folder, f"model_{epoch}.h5"))
        
        #######################################################################################
        # VALIDATION
        #######################################################################################
        val_metrics = list()
        for client in clients:
            valid_loss, valid_error, valid_label_storage, valid_pred_storage, valid_cm, model = \
                setup(client.valid_class_distribution.shape[0], client.model)
            for dpt_idx, datapoint_batch in enumerate(client.valid_data.take(client.valid_steps_per_epoch)):
                prediction_batch, loss_batch = valid_step(client.model, datapoint_batch, client.class_weights, label_type)
                valid_loss = valid_loss + np.mean(loss_batch)
                update_standard_metrics(client.model, datapoint_batch, prediction_batch, valid_batch_size)

                if (len([metric.name for metric in client.model.metrics if metric.name in ['cohens_kappa', 'c_index_censor']])) > 0:
                    handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                                            client.model, dpt_idx, client.train_steps_per_epoch, label_type)

            client.valid_metrics = {**{'_'.join(('val', m.name)): m.result() for m in client.model.compiled_metrics._metrics}, 'val_loss': valid_loss/(dpt_idx+1)}
            
            if train_params['monitor_val'] in client.valid_metrics.keys():
                metr = train_params['monitor_val']
            else:
                for i in range(10):
                    if f"{train_params['monitor_val']}_{i}" in client.valid_metrics.keys():
                        metr = f"{train_params['monitor_val']}_{i}"
            val_metrics.append(client.valid_metrics[metr])

            # update callbacks
            for callback in client.callbacks:
                callback.on_epoch_end(epoch, {**client.train_metrics, **client.valid_metrics})
                

            print(f'# {client.name} #')
            print('Training -   ')
            for v in client.train_metrics:
                print('    {:s}: {:.4f}   '.format(v, client.train_metrics[v]), end='')
            print('')
            
            print('Validation -   ')
            for v in client.valid_metrics:
                print('    {:s}: {:.4f}   '.format(v, client.valid_metrics[v]), end='')
            print('')
        
        ### get best model based on all accs.
        current_metric = np.mean(val_metrics)
        if "loss" in train_params['monitor_val']:
            if current_metric < best_metric:
                best_epoch = epoch
                best_metric = current_metric
        else:
            if current_metric > best_metric:
                best_epoch = epoch
                best_metric = current_metric
        # clean up old models to save space
        for client in clients:
            if best_epoch == epoch:
                for i in range(0, epoch):
                    model_file = os.path.join(client.config['model_save_path'], str(_run._id), f"model_{i}.h5")
                    if os.path.isfile(model_file):
                        os.remove(model_file)
            else:
                model_file = os.path.join(client.config['model_save_path'], str(_run._id), f"model_{epoch}.h5")
                if os.path.isfile(model_file):
                    os.remove(model_file)

        print(f'Time used for this epoch: {round(time.time()-s, 0)} ')

        # check if early stopping applies
        if client.model.stop_training:
            break

    #######################################
    # AT END OF TRAINING - LOAD BEST MODEL
    #######################################
    print(f'Overall time for training: {round(time.time()-training_start, 0)} seconds')
    print(f"best epoch: {best_epoch}")
    server = clients[0]

    _log.debug('Finished training -> evaluation')

    server.model.load_weights(os.path.join(server.results_folder, f"model_{best_epoch}.h5"))
    best_weights = server.model.get_weights()

    _log.debug('Best epoch: {}'.format(best_epoch))

    best_result = np.max(server.callbacks[-1].history[train_params['monitor_val']])
    _log.debug('best {monitor_val}: {result}'.format(monitor_val=train_params['monitor_val'], result=best_result))

    return best_weights, server.callbacks[-1]


