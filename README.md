# Federated ISUP prediction
This repository contains all scripts to train and evaluate a federated model for ISUP grading.
To run the lightweight versions including embeddings from foundation models see folder `foundation_models`.
The model requires histopathology images (TMA spots) as input and predicts the corresponding ISUP grade.

Directly go to the foundation models [README file](https://github.com/LohmannJens/federated-ISUP/blob/main/foundation_models/README.md).

## Setup
To run the code multiple python libraries need to be installed.
A conda environment is included in this repository `environment.yml`.
With conda installed on the machine the following command needs to be executed:
```
conda env create -f environment.yml
conda activate fed_isup
```
We used [sacred](https://sacred.readthedocs.io/en/stable/) to store all experiments.
For each training run, a folder with an increasing id will be created automatically and all information about the run will be stored in that folder.
Training can be started by setting the working directory to `run_files/` and using the following command:
```
python train_model.py with ../testdata/test_config.yaml 
```


## Data preprocessing
The script to perform the image preprocessing `cut_center_pieces.py` can be found in folder `preparation/ `
The image metainformation need to be stored in a csv file with the image path and the ISUP grades as columns.
You need separate csv files for your training, validation and test sets. Here is an example:

img_path | isup |
---------|------|
img1.png |0     |
img2.png |4     |
img3.png |2     |


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

There is one config file `test_config.yaml` to define the overall training process (e.g. epochs, learning rate, etc.).
Details can be found in the general `config.yaml`.
Additionally, a pretrained Inception-v3 needs to be downloaded for the desired patch image size and saved in `models/`.
For the test data it is already included in this repository.
For each client the data is split in individual csv files for train, validation, and testing.
The preprocessed images are stored in the folder `images/` as PNG.


## Lightweight training with foundation model embeddings
All scripts to train the architectures based on the foundation model embeddings are in the folder `foundation_models`.
Please refer to the additonal [README file](https://github.com/LohmannJens/federated-ISUP/blob/main/foundation_models/README.md) for further explanations.
This part of the repository is implemented in PyTorch to be in line with the implementations of UNI and CONCH.
Therefore, the folder `foundation_models` includes an additional conda environment that can be installed using:

```
conda env create -f environment.yml
conda activate isuptorch
```

To create the embeddings of UNI and CONCH please refer to the implementations of the original papers.
An example embedding for each client for the two foundation models is given in the folder `testdata/data`.


## Additional scripts
Additonal scripts are included in this repository for further evaluations.
This includes the generation of the baseline data and scripts to generate the final figures of the publication.
They can be found in the folder `scripts/`.
