# Foundation model ISUP prediction
This folder contains all scripts to train and evaluate a federated model for ISUP grading using foundation models.
The model requires histopathology images (TMA spots) as input and predicts the corresponding ISUP grade from embeddings created with foundation models (e.g. UNI, CONCH).


## Setup
To run the code multiple python libraries need to be installed.
A conda environment is included in this repository `environment.yml`.
With conda installed on the machine the following command needs to be executed:
```
conda env create -f environment.yml
conda activate isuptorch
```

Training can be started by using the following command:
```
python train_model.py
```

## Test data
We included a few test samples to show the setup for the federated learning scheme.
The folder has the following structure:
```
testdata
│   test_config.yaml
└───data
    └───client1
    |   │   client_config.yaml
    |   │   test.csv
    |   │   train.csv
    |   │   valid.csv
    |   │   conch_embeddings.csv
    |   │   uni_embeddings.csv
    |   └───images
    |       |    *.png
    |       |    ...
    |
    └───client2
        │   client_config.yaml
        │   test.csv
        │   train.csv
        │   valid.csv
        │   conch_embeddings.csv
        │   uni_embeddings.csv
        └───images
            |    *.png
            |    ...
```

There are two config file to define the overall training process (e.g. epochs, learning rate, etc.) for the two foundation models UNI and CONCH.
For each client the data is split in train, validation, and testing while running the script.


## Creation of model embeddings
To create the embeddings of UNI and CONCH please refer to the implementations of the original papers.
An example embedding for each client for the two foundation models is given in the folder `testdata/data`.
