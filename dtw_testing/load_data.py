import numpy as np
import os
from difflib import get_close_matches
from pathlib import Path

def load_dataset(name):
    this_path = Path(__file__)
    data_path = Path(this_path).parent / 'DATA'
    train_path = data_path / name / f'{name}_TRAIN.tsv'
    test_path = data_path / name / f'{name}_TEST.tsv'


    if not os.path.isfile(train_path):
        datasets = [p.stem for p in data_path.iterdir()]
        close_match = get_close_matches(name, datasets, n=1)

        if len(close_match) > 0:
            raise ValueError(f'No dataset named {name}, did you mean {close_match[0]}')
        else:
            raise ValueError(f'No dataset named {name}')

    train_data = np.genfromtxt(fname=train_path, delimiter="\t", skip_header=0, filling_values=np.nan)
    test_data = np.genfromtxt(fname=test_path, delimiter="\t", skip_header=0, filling_values=np.nan)
    return train_data, test_data

def generate_query(data=..., dataset=..., train=False, seed=...):
    if dataset is not Ellipsis:
        data = load_dataset(dataset, train)

    if seed is not Ellipsis:
        np.random.seed(seed)

    mask = np.array([False] * len(data))
    mask[np.random.randint(len(data))] = True

    return data[mask].flatten(), data[~mask]