import glob
import torch
from torch.utils.data import Dataset
from typing import List
import tqdm

def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from. 

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list




def convert_line2idx(line, vocab):
    """ Converts a string into a list of character indices
    Input
    ------------
    line: str. A line worth of data
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    -------------
    line_data: List[int]. List of indices corresponding to the characters
                in the input line.
    """
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data




def convert_files2idx(files, vocab):
    """ This method iterates over files. In each file, it iterates over
    every line. Every line is then split into characters and the characters are 
    converted to their respective unique indices based on the vocab mapping. All
    converted lines are added to a central list containing the mapped data.
    Input
    --------------
    files: List[str]. List of files in a particular split
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    ---------------
    data: List[List[int]]. List of lists where each inner list is a list of character
            indices corresponding to a line in the training split.
    """
    data = []

    for file in files:
        with open(file, encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            toks = convert_line2idx(line, vocab)
            data.append(toks)

    return data


class characterDataset(Dataset):
    def __init__(self, data: List[tuple]):
        print("Initializing Dataset")
        self.x = []
        self.y = []
        for line in tqdm.tqdm(data):
            self.x.append(line[0])
            self.y.append(line[1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y