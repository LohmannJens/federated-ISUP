import logging

import numpy as np
import tensorflow as tf

import training_evaluation.model_losses as own_losses
from training_evaluation.model_metrics import TfCategoricalAccuracy


def load_base_model(model_config):
    """
    either load a model predefined in tf.keras (pretrained or not) or load an own model
    :param base_model:
    :param model_config:
    :return:
    """
    try:
        bmodel = tf.keras.models.load_model(model_config['base_model'])
    except:
        print(f"Model {model_config['base_model']} could not be loaded.")
        
    if 'cut_off_layer' in model_config:
        cut_off_after_layer = model_config['cut_off_layer']
    else:
        cut_off_after_layer = False
    if isinstance(cut_off_after_layer, int) and not isinstance(cut_off_after_layer, bool):
        x = bmodel.layers[cut_off_after_layer].output
    elif isinstance(cut_off_after_layer, str):
        x = bmodel.get_layer(cut_off_after_layer).output
    else:
        x = bmodel.layers[-1].output

    for layer in bmodel.layers:
        layer._name = layer._name + str('_base')

    return bmodel, x


def compile_model(model, train_params, metric_class_weights=None, _log=logging):
    """

    :param model: tensorflow/keras model
    :param train_params: dictionary with 'optimizer.name'
    :param metric_class_weights:
    :param _log:
    :return:
    """
    try:
        # parameters are a list, so not unintentionally overwritten by main config
        optimizer = getattr(tf.keras.optimizers, train_params["optimizer"]['name'])(
                **{k: v for d in train_params['optimizer']['params'] for k, v in d.items()})
    except AttributeError:
        raise NotImplementedError("only optimizers available at tf.keras.optimizers are implemented at the moment")

    try:
        loss = getattr(tf.keras.losses,  train_params["loss_fn"])
    except AttributeError:
        try:
            loss = getattr(own_losses, train_params['loss_fn']+'_wrap')()
        except AttributeError:
            raise NotImplementedError("only losses "
                                      "available at tf.keras.losses or "
                                      "cdor, deepconvsurv and ecarenet_loss "
                                      "are implemented at the moment")
    except TypeError:
        print('loss_information', *train_params['loss_fn'])
        loss = getattr(own_losses, train_params['loss_fn'][0]+'_wrap')(*train_params['loss_fn'][1:])

    # read all metrics from the list in config.yaml
    metrics = []
    if train_params['compile_metrics'] is not None:
        for metric in train_params["compile_metrics"]:
            try:
                metrics.append(getattr(tf.keras.metrics, metric)())
            except AttributeError:
                try:
                    if metric == "TfCategoricalAccuracy":
                        metrics.append(TfCategoricalAccuracy(metric_class_weights))
                except AttributeError:
                    raise NotImplementedError("the given metric {} is not implemented!".format(metric))


    compile_attributes = train_params["compile_attributes"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        **compile_attributes
    )

    if _log is not None:
        _log.info("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    else:
        print("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    return model


def get_class_weights(class_weight, metric_weight, class_distribution, _log):
    if class_weight:
        class_weights = {i: sum(class_distribution)/class_distribution[i] if class_distribution[i] != 0 else 0
                         for i in range(len(class_distribution))}
        max_weight = np.max([class_weights[k] for k in class_weights])
        class_weights = {k: class_weights[k]/max_weight for k in class_weights}
        _log.debug('class weights: ')
        _log.debug(class_weights)
    else:
        class_weights = None
    if metric_weight:
        metric_class_weights = class_weights
    else:
        metric_class_weights = None
    return class_weights, metric_class_weights
