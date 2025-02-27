import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset


class ISUPPredictor(nn.Module):
    """
        Predictor layers of the eCaReNet M_ISUP module

        :param input_channels: number of input features, equals embedding space
        :param n_classes: number of classes to predict (6 for ISUP)
    """
    def __init__(self, input_channels, n_classes=6):
        super(ISUPPredictor, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(input_channels)
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dense_layers.append(nn.Linear(input_channels, 32))
        self.batch_norms.append(nn.BatchNorm1d(32))
        self.output_layer = nn.Linear(32, n_classes - 1)
    
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        for dense, bn in zip(self.dense_layers, self.batch_norms):
            x = F.relu(bn(dense(x)))
        x = torch.sigmoid(self.output_layer(x))
        return x


class ISUPDataset(Dataset):
    """
        Dataset class to use with the ISUPPredictor class.
        Loads the embeddings as features (X) and ISUP classes as labels (y)

        :param df: Pandas DataFrame containing the following rows:
            - img_path: Path to each image, which is treated as an unique ID
            - isup:     ISUP score as int (0-5)
            - others:   All other rows are considered as part of the embedding.
                        Additonal information needs to be filtered (df.drop()) before usage.
    """
    def __init__(self, df):
        self.img_paths = df["img_path"].values
        self.X = df.drop(columns=["isup", "img_path"]).values
        self.y = df["isup"].values
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ISUPCCELoss(nn.Module):
    """
        PyTorch implementation of the special Cross-Entropy Loss used for eCaReNet.
        The ISUP score is defined by a vector of length 5 with each position containing
        an float value between 0.0 and 1.0. The sum over all 5 positions is the final ISUP
        grade. This introduces a smaller numerical distance between more simimlar scores.
        
    """
    def __init__(self):
        super(ISUPCCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        result = torch.where(y_true == 1.0, y_pred, 1.0 - y_pred)
        result_clipped = torch.clamp(result, min=1e-7, max=1 - 1e-3)
        log_result = torch.log(result_clipped)
        sum_log_result = -torch.sum(log_result, dim=1)
        return sum_log_result.sum()


class Client():
    """
        Client class used for trainig.

        :param client_name: used for naming the output folders
        :param train_loader: pytorch dataloader of the training dataset
        :param val_loader: pytorch dataloader of the validation dataset
        :param model: ISUPPredictor model of the client
        :param lr: learning rate used for training the model
        :param device: device used for training (CPU, GPU, etc.)
    """
    def __init__(self, client_name, train_loader, val_loader, model, lr, device):
        self.client_name = client_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.n_train_data = len(self.train_loader.dataset)
        self.train_metrics = dict({"acc": list(), "loss": list(), "val_acc": list(), "val_loss": list()})

        self.criterion = ISUPCCELoss().to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=lr)