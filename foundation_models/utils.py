import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes import ISUPDataset
from collections import OrderedDict
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def get_dataset(filename, embedding_df):
    """
        Function that combines the embeddings of UNI/CONCH with the isup
        scores and IDs (img_path).

        :param filename: location where the meta data is stored
        :param embedding_df: DataFrame() with the embeddings (UNI/CONCH)

        :return: Embeddings and metadata combined in ISUPDataset() instance
    """
    df = pd.read_csv(filename)
    merged_df = pd.merge(embedding_df, df[['img_path', 'isup']], on='img_path', how='inner')
    dataset = ISUPDataset(merged_df)
    return dataset


def output_to_label(tensor):
    """
        converts the eCaReNet representation [1,1,1,0,0] to ISUP label [3].
        Reverses label_to_output()
        
        :param tensor: eCaReNet specific representation of the ISUP scores as torch.tensor
                       can be multidimensional for batches.

        :return: list of ISUP scores as int
    """
    return torch.round(torch.sum(tensor, dim=1)) 


def label_to_output(tensor):
    """
        converts ISUP label [3] to eCaReNet representation [1,1,1,0,0].
        Reverses output_to_label()

        :param tensor: list of ISUP scores as int
        
        :return: eCaReNet specific representation of the ISUP scores as torch.tensor
    """
    num_rows = tensor.size(0)
    result = torch.zeros((num_rows, 5), dtype=torch.float)
    for i, n in enumerate(tensor):
        result[i, :n] = 1
    return result


def fed_avg(state_dicts, n_data_clients):
    """
        Basic Federated Averaging method. Average across the given models by
        weighting each model by the number of its datapoints.

        :param state_dicts: List of the model states
        :param n_data_clients: List of number of data points per client

        :return: aggregated model while weighting by number of data points
    """
    avg_state_dict = OrderedDict()
    total_data_points = sum(n_data_clients)
    for key in state_dicts[0].keys():
        for i, (sd, n_client) in enumerate(zip(state_dicts, n_data_clients)):
            if i == 0:
                avg_state_dict[key] = sd[key] * (n_client / total_data_points)    
            else:
                avg_state_dict[key] += sd[key] * (n_client / total_data_points)

    return avg_state_dict


def plot_training_metrics(clients, resultspath, id):
    """
        Function to plot the trainig and valudation metrics after training.

        :param clients: list of Client() objects that participated the training
        :param resultspath: file path to store the results
        :param id: indicates the number of the run (for training multiple rounds at once)

        :return: None
    """
    def create_metric_plot(data, name, client_name, resultspath, id):
        figurepath = os.path.join(resultspath, client_name, f"{id}_{name}.png")
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(data)), data, linestyle='-', color='b')
        plt.title(client_name)
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.grid(True)
        plt.savefig(figurepath)
        plt.close()

    for c in clients:
        for k, v in c.train_metrics.items():
            create_metric_plot(v, k, c.client_name, resultspath, id)



def evaluate(model, dataloader, resultspath, id):
    """
        Evaluate the model on the given dataset.

        :param model: the model to evaluate
        :param dataloader: The testing dataset to evaluate
        :param resultspath: file path to store the results
        :param id: indicates the number of the run (for training multiple rounds at once)

        :return: dictionary containing accuracy, Cohen's kappa, and F1-score.
    """
    # Set model to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(inputs)
            scores_pred = output_to_label(outputs)
            scores_true = labels

            # Collect predictions and labels
            all_preds.extend(scores_pred.cpu().numpy())
            all_labels.extend(scores_true.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='horizontal')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(os.path.join(resultspath, f"{id}_confusion_matrix.png"))
    plt.cla()
    plt.close(disp.figure_)

    print(f"acc.:\t\t{accuracy}")
    print(f"kappa:\t\t{kappa}")
    print(f"f1-score:\t{f1}")

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "f1-score": f1
    }
