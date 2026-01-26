import os
import sys
import yaml
import torch

import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.append("..")
from classes import ISUPDataset, Client, ISUPPredictor
from utils import get_dataset, evaluate, fed_avg
from train_model import train, validate, plot_training_metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_single_central(client_name, other_cohorts, embedding_df, model):
    df = pd.concat(other_cohorts)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['isup'])
    merged_df = pd.merge(embedding_df, train_df[['img_path', 'isup']], on='img_path', how='inner')
    train_dataset = ISUPDataset(merged_df)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    
    merged_df = pd.merge(embedding_df, val_df[['img_path', 'isup']], on='img_path', how='inner')
    val_dataset = ISUPDataset(merged_df)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    clients = [Client(client_name, train_loader, val_loader, model, config["lr"], device)]

    testing = [val_df]

    return clients, testing


def prepare_central(client_name, other_cohorts, single_cohort_df, embedding_df, model):
    df = pd.concat(other_cohorts)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['isup'])
    merged_df = pd.merge(embedding_df, train_df[['img_path', 'isup']], on='img_path', how='inner')
    train_dataset = ISUPDataset(merged_df)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    
    merged_df = pd.merge(embedding_df, val_df[['img_path', 'isup']], on='img_path', how='inner')
    val_dataset = ISUPDataset(merged_df)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    clients = [Client(client_name, train_loader, val_loader, model, config["lr"], device)]

    testing = [single_cohort_df]

    return clients, testing


def prepare_federated(client_name, other_cohorts, single_cohort_df, path, embedding_df, model):
    clients = list()
    for coh_id, coh in enumerate(other_cohorts):
        train_df, val_df = train_test_split(coh, test_size=0.2, stratify=coh['isup'])
        merged_df = pd.merge(embedding_df, train_df[['img_path', 'isup']], on='img_path', how='inner')
        train_dataset = ISUPDataset(merged_df)
        train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
                        
        merged_df = pd.merge(embedding_df, val_df[['img_path', 'isup']], on='img_path', how='inner')
        val_dataset = ISUPDataset(merged_df)
        val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
                        
        os.makedirs(os.path.join(path, f"{client_name}_client{coh_id}"), exist_ok=True)
        clients.append(Client(f"{client_name}_client{coh_id}", train_loader, val_loader, model, config["lr"], device))

    testing = [single_cohort_df]
    return clients, testing


def prepare_local(client_name, other_cohorts, single_cohort_df, embedding_df, model):
    train_df, val_df = train_test_split(single_cohort_df, test_size=0.2, stratify=single_cohort_df['isup'])
    merged_df = pd.merge(embedding_df, train_df[['img_path', 'isup']], on='img_path', how='inner')
    train_dataset = ISUPDataset(merged_df)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
                    
    merged_df = pd.merge(embedding_df, val_df[['img_path', 'isup']], on='img_path', how='inner')
    val_dataset = ISUPDataset(merged_df)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    clients = [Client(client_name, train_loader, val_loader, model, config["lr"], device)]
     
    testing = other_cohorts
    return clients, testing


def run_training(clients, config, testing, resultspath, results, mode, id, idx, embedding_df):
    num_epochs = config["epochs"]
    best_epoch = 0
    best_acc = 0.0
    for epoch in range(num_epochs):
        print_to_stdout = True if epoch % (num_epochs/5) == 0 else False
        print_to_stdout = False # just for now for running multiple experiments with refined parameters

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
            torch.save(clients[0].model.state_dict(), "temp_model_weights.pth")

    print(f"\nBest epoch: {best_epoch}\t with val. acc.:{best_acc}")
    plot_training_metrics(clients, resultspath, id)

    final_model = clients[0].model
    final_model.load_state_dict(torch.load("temp_model_weights.pth"))
    final_model.eval()

    # Final testing against external dataset
    print()
    print("TESTING")
    for test_df in testing:
        testing_name = f"{id}_{mode}_{idx}"
        print(testing_name)

        merged_df = pd.merge(embedding_df, test_df[['img_path', 'isup']], on='img_path', how='inner')
        test_dataset = ISUPDataset(merged_df)    
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
        res_dict = evaluate(final_model, test_loader, resultspath, f"{id}_{testing_name}")
                        
        res_dict["best_epoch"] = best_epoch
        res_dict["test_dataset"] = f"{mode}_{idx}"
        results.append(res_dict)

    print()
    print("TMA20")
    pro13_embedding_path = os.path.join("..", "embeddings", f"{config['model_name']}_Pro13.1D_embeddings.csv")
    pro13_embedding_df = pd.read_csv(pro13_embedding_path).T.reset_index().rename(columns={'index': 'img_path'})
    pro13_dataset = get_dataset("/data/data_split_gleasonaut/Pro13.1D.csv", pro13_embedding_df)
    pro13_loader = DataLoader(pro13_dataset, batch_size=500, shuffle=False)
    pro13_res_dict = evaluate(final_model, pro13_loader, resultspath, f"{id}_pro13")

    for r_dict in [results[-1]]:
        r_dict["accuracy_TMA20"] = pro13_res_dict["accuracy"]
        r_dict["kappa_TMA20"] = pro13_res_dict["kappa"]
        r_dict["f1-score_TMA20"] = pro13_res_dict["f1-score"]

    return results


def start(config, n_clients):
    df = pd.read_csv(os.path.join("data", "full.csv"))

    def stratified_split(df, n_splits):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list()
        for _, test_index in skf.split(df, df["isup"]):
            splits.append(df.iloc[test_index])
        return splits

    if n_clients > 1:
        cohorts = stratified_split(df, n_clients)
    else:
        cohorts = [df]

    # load embeddings
    embedding_path = os.path.join("..", "embeddings", f"{config['model_name']}_full_embeddings.csv")
    embedding_df = pd.read_csv(embedding_path).T.reset_index().rename(columns={'index': 'img_path'})

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", config["model_name"]), exist_ok=True)
    resultspath = os.path.join("results", config["model_name"], f"{n_clients}_clients")
    os.makedirs(resultspath, exist_ok=True)

    results = list()
    repeats_dict = dict({1:40, 3:13, 5:8, 10:4, 20:2, 40:1})
    repeats = repeats_dict[n_clients]
    for id in range(repeats):
        for idx in range(n_clients):
            # if just one client is there only run central (because all are the same)
            if n_clients == 1:
                modes = ["central"]
                other_cohorts = cohorts
            else:
                modes = ["central", "federated", "local"]
                single_cohort_df = cohorts[idx]
                if idx == 0:
                    other_cohorts = cohorts[idx+1:]
                elif idx == n_clients-1:
                    other_cohorts = cohorts[:idx]
                else:
                    other_cohorts = cohorts[:idx] + cohorts[idx+1:]

            # start here the central, federated and local training
            for mode in modes:
                model = ISUPPredictor(input_channels=config["input_channels"], n_classes=6)
                model.to(device)

                client_name = f"{id}_{mode}_{idx}"
                if mode != "federated":
                    os.makedirs(os.path.join(resultspath, client_name), exist_ok=True)

                # initialize clients
                if n_clients == 1:
                    clients, testing = prepare_single_central(client_name, other_cohorts, embedding_df, model)
                else:
                    if mode == "central":
                        clients, testing = prepare_central(client_name, other_cohorts, single_cohort_df, embedding_df, model)
                    elif mode == "federated":
                        clients, testing = prepare_federated(client_name, other_cohorts, single_cohort_df, resultspath, embedding_df, model)
                    elif mode == "local":
                        clients, testing = prepare_local(client_name, other_cohorts, single_cohort_df, embedding_df, model)
                        
                # start training
                results = run_training(clients, config, testing, resultspath, results, mode, id, idx, embedding_df)

    return results



for config_file in ["uni.yml", "conch.yml"]:
#for config_file in ["test.yml"]:
    with open(os.path.join("..", "configs", config_file), "r") as f:
        config = yaml.safe_load(f)
        for n_clients in [1, 3, 5, 10, 20, 40]:
            results = start(config, n_clients)
            final_df = pd.DataFrame(results)
            filename = f"V2_{config['model_name']}_{n_clients}_increasing_clients.csv"
            final_df.to_csv(os.path.join("results", filename), index=False)