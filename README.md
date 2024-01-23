# Periodic Set Transformer

This is a repository for the implementation of the Periodic Set Transformer. The starter code was taken from CGCNN:
https://github.com/txie-93/cgcnn. The general usage is largely the same as their model.

## File Descriptions
- `main.py`: Main file for command line utility
- `model.py`: Contains the full model implementation for the PST
- `matbench_parameters.py` Defines the parameters used in the Matbench experiments
- `mb.py`: Main file for running the Matbench test suite
- `data.py`: Defines all PyTorch datasets used in `main.py` and `mb.py`
- `train.py`: Implements the training and validation cycles
- `plots.py`: Defines the functions for creating the plots in the article
- `gpr.py`: The Gaussian Process Regression implementation used in the experiments
- `pdd_helpers.py` - Helper functions for creating PDDs from CIFs
- `atom_init.json`: Defines CGCNN atom features in a one-hot encoded manner
- `mat2vec.csv`: Defined Mat2Vec atom features
- `mf`: This folder contains the Jarvis IDs that were in the train, validation and test set when Matformer was run
- `jarvis_dft_2d_2021_pymatgen_structures.zip`: Zip file containing the data to run the Jarvis-DFT dataset

## General Usage

First, install the dependencies via `pip install -r requirements.txt`.

To run the model there needs to be `data` folder containing the following:

1. A set of [CIF] files with the following names convention `id.cif`
2. A [CSV] files named `id_prop.csv` with two columns: the first is the name of the corresponding [CIF] filename without the extension and the second is the property value

Please make sure that the atom_init.json file is in the current directory along with `main.py`.

The model can then be run by using the following:

```bash
python main.py data
```

A number of model options are listed in `main.py` that can be adjusted as needed. For example:

```bash
python main.py --epochs=200 data
```

## Matbench

In order to run the test suite for Matbench please run the following:
`pip install -r requirements.txt` to install the necessary packages, then
`python mb.py`.

The `results.tar.gz` file will appear in the same directory.

## Jarvis-DFT

To run the Jarvis-DFT dataset first, unzip the data file `jarvis_dft_3d_pymatgen_structure.zip`. 
The resulting binary will automatically be read in after running `python run_jarvis.py`.