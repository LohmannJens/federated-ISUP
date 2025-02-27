import os
import yaml
import torch
import warnings

from torch.utils.data import DataLoader
import pandas as pd

from classes import ISUPPredictor, Client
from utils import get_dataset, output_to_label, label_to_output, fed_avg, plot_training_metrics, evaluate

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_clients(datapath, resultspath, model, config, device):
    """
        Function to initialize the clients and make them ready for training.

        :param datapath: filepath to the underlying data (each client as one subfolder)
        :param resultspath: filepath to store the trainig results
        :param model: Model to be trained, is a ISUPPredictor() instance
        :param config: config file with important attributes for the training
        :param device: device used for training (CPU, GPU, etc.)

        :return: list of Client() instances participating in the training
    """
    clients = list()
    for folder_name in os.listdir(datapath):
        folder_path = os.path.join(datapath, folder_name)
        
        os.makedirs(os.path.join(resultspath, folder_name), exist_ok=True)  
        embedding_path = os.path.join(folder_path, f"{config['model_name']}_embeddings.csv")
        embedding_df = pd.read_csv(embedding_path).T.reset_index().rename(columns={"index": "img_path"})

        train_dataset = get_dataset(os.path.join(folder_path, "train.csv"), embedding_df)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        val_dataset = get_dataset(os.path.join(folder_path, "valid.csv"), embedding_df)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        clients.append(Client(folder_name, train_loader, val_loader, model, config["lr"], device))
    return clients


def train(clients, device, print_to_stdout):
    """
        Training routine for the clients.
    
        :param clients: list of Client() instances participating in the training
        :param device: device used for training (CPU, GPU, etc.)
        :param print_to_stdout: Bool value, helps to print metrics only every n epochs

        :return: None
    """
    for c in clients:
        c.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in c.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            c.optimizer.zero_grad()
            outputs = c.model(inputs)

            full_pred = outputs
            full_true = label_to_output(targets).to(device)
            scores_pred = output_to_label(outputs).to(device)
            scores_true = targets

            loss = c.criterion(full_pred, full_true)
            loss.backward()
            c.optimizer.step()

            running_loss += loss.item()
            total += targets.size(0)
            correct += (scores_pred == scores_true).sum().item()

        epoch_loss = running_loss / len(c.train_loader.dataset)
        epoch_accuracy = 100 * correct / total

        if print_to_stdout:
            print(f"\t{c.client_name} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        c.train_metrics["acc"].append(epoch_accuracy)
        c.train_metrics["loss"].append(epoch_loss)


def validate(clients, device, print_to_stdout):
    """
        Validation routine for the clients.
    
        :param clients: list of Client() instances participating in the training
        :param device: device used for training (CPU, GPU, etc.)
        :param print_to_stdout: Bool value, helps to print metrics only every n epochs

        :return: overall validation accuracy across all clients
    """
    overall_acc = 0
    for c in clients:
        c.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in c.val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = c.model(inputs)

                full_pred = outputs
                full_true = label_to_output(targets).to(device)
                scores_pred = output_to_label(outputs).to(device)
                scores_true = targets

                loss = c.criterion(full_pred, full_true)
                val_loss += loss.item()
                total += targets.size(0)
                correct += (scores_pred == scores_true).sum().item()

        val_loss = val_loss / len(c.val_loader.dataset)
        val_accuracy = 100 * correct / total
        if print_to_stdout: 
            print(f"\t{c.client_name} Val. Loss: {val_loss:.4f}, Val. Accuracy: {val_accuracy:.2f}%")
            
        c.train_metrics["val_acc"].append(val_accuracy)
        c.train_metrics["val_loss"].append(val_loss)
        overall_acc += val_accuracy
    
    return overall_acc / len(clients)


def run_training(datapath, resultspath, device, id, config):
    """
        Runs the main training rountine. The differentiation between federated,
        central, and local training comes by the setup of the datapath.

        :param datapath: filepath to the underlying data (each client as one subfolder)
        :param resultspath: filepath to store the trainig results
        :param device: device used for training (CPU, GPU, etc.)
        :param id: indicates the number of the run (for training multiple rounds at once)
        :param config: config file with important attributes for the training

        return: dictionary including model performance on testing datset
    """
    # initialize model, loss, optimizer for training
    model = ISUPPredictor(input_channels=config["input_channels"], n_classes=6)
    model.to(device)
    
    # initialize the clients
    clients = initialize_clients(datapath, resultspath, model, config, device)

    # start trainig
    num_epochs = config["epochs"]
    best_epoch = 0
    best_acc = 0.0
    for epoch in range(num_epochs):
        print_to_stdout = True if epoch % (num_epochs/5) == 0 else False
        if print_to_stdout:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        # training
        train(clients, device, print_to_stdout)
        
        # Use FedAvg for aggregation
        n_data_clients = list()
        client_states = list()
        for c in clients:
            n_data_clients.append(c.n_train_data)
            client_states.append(c.model.state_dict())
        server_state = fed_avg(client_states, n_data_clients)
        for c in clients:
            c.model.load_state_dict(server_state)

        # validation
        overall_acc = validate(clients, device, print_to_stdout)

        if best_acc < overall_acc:
            best_epoch = epoch
            best_acc = overall_acc
            torch.save(model.state_dict(), os.path.join(resultspath, f"{id}_model_weights.pth"))

    print(f"\nBest epoch: {best_epoch}\t with val. acc.:{best_acc}")
    plot_training_metrics(clients, resultspath, id)

    final_model = clients[0].model
    final_model.load_state_dict(torch.load(os.path.join(resultspath, f"{id}_model_weights.pth")))
    final_model.eval()

    # Final testing against external dataset
    print()
    print("TESTING")
    results = list()
    for folder_name in os.listdir(datapath):
        folder_path = os.path.join(datapath, folder_name)
        
        os.makedirs(os.path.join(resultspath, folder_name), exist_ok=True)  
        embedding_path = os.path.join(folder_path, f"{config['model_name']}_embeddings.csv")
        embedding_df = pd.read_csv(embedding_path).T.reset_index().rename(columns={"index": "img_path"})

        test_dataset = get_dataset(os.path.join(folder_path, "test.csv"), embedding_df)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


        res_dict = evaluate(final_model, test_loader, resultspath, f"{id}_{folder_name}")
            
        res_dict["best_epoch"] = best_epoch
        res_dict["test_dataset"] = folder_name
        results.append(res_dict)

    return results


def loop_training(repeats, config):
    """
        Runs the trainings multiple times (indicated by repeats parameter).
        Returns then the combined results.
        For testing just set repeats=1.
        To differentiate between federated, local, and centralized training the
        datapaths need to be set accordingly.

        :param repeats: indicates the number of times to run the training
        :param config: config file with important attributes for the training

        :return: pandas DataFrame including the model performances
    """
    os.makedirs("results", exist_ok=True)
    resultspath = os.path.join("results", config["model_name"])
    os.makedirs(resultspath, exist_ok=True)

    results_list = list()
    for id in range(repeats):
        print()            
        results_list.extend(run_training(config["main_datapath"], resultspath, device, id, config))

    df = pd.DataFrame(results_list)
    return df
    

def main():
    # this can be adapted for training multiple times
    repeats = 1

    for config_file in ["uni.yml", "conch.yml"]:
        with open(os.path.join("configs", config_file), "r") as f:
           config = yaml.safe_load(f)

        results = list()
        print()
        print(config["model_name"])
        results.append(loop_training(repeats, config))

        final_df = pd.concat(results)    
        filename = f"{repeats}_{config['model_name']}_results.csv"
        resultspath = os.path.join("results", filename)
        final_df.to_csv(resultspath, index=False)


if __name__ == "__main__":
    main()