import os

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from settings.default_settings import (EXPERIMENTS_DIR, DEFAULT_CONFIG_FILE)

# define sacred expirment (ex) and add data_ingredientdocker
ex = Experiment('fed-isup', interactive=True, base_dir=os.path.abspath(os.path.join(os.getcwd(), '..')))
ex.add_config(DEFAULT_CONFIG_FILE)


if not os.path.isdir(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)
ex.observers.append(FileStorageObserver.create(EXPERIMENTS_DIR))

ex.capture_out_filter = apply_backspaces_and_linefeeds
