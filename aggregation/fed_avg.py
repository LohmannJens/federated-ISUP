import numpy as np

def fed_avg(models, n_datapoints):
    """
    Aggregates the weights of the models of each clients to generate a generalised model.
    Uses the simplest approach for this by calculating the unweighted mean of each weight.
    
    :param models: list of all models used 
    :return: aggregated weights
    """
    all_datap = sum(n_datapoints)
    weights = list([list() for _ in range(len(models[0].get_weights()))])
    for model, n_datap in zip(models, n_datapoints):
        for i, w in enumerate(model.get_weights()):
            weights[i].append(w * (n_datap/all_datap))
    final_weights = [np.array(sum(w_l)) for w_l in weights]

    return final_weights

