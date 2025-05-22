## CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

### Data Exploration and Set-up
- All data (including precomputed features and pretrain models) can be downloaded from <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench>
- A self-contained tutorial on data download, data exploration, and baseline training: [`data_tutorial.ipynb`](data_tutorial.ipynb)
- Additional details can be found in <https://cosmobench.streamlit.app/>
- To run all baselines:
  - set up an virtual enviroment
    ```
    python3 -m venv venv
    source venv/bin/activate  
    ```
  - install dependencies: `pip install -r requirements.txt`

### Task 1: Predicting Cosmological Parameters from Point Clouds
- Data preparation: See details in [`models/cloud_param`](./models/cloud_param/)
- Cosmological baseline (2PCF): `python models/cloud_param/2PCF.py`
- Linear Least Squares (LLS): `python models/cloud_param/simple_param.py`
- Graph Neural Networks (GNNs): See implementations in <https://github.com/Byeol-Haneul/CosmoTopo/tree/benchmark>

### Task 2: Predicting Velocities from Point Positions
- Data preparation: See details in [`models/cloud_velocity`](./models/cloud_velocity/)
- Cosmological baseline (linear theory): `python models/cloud_velocity/linear_theory.py`
- LLS: `python models/cloud_velocity/simple_velocity.py`
- GNNs: `python train_velocity.py`

### Task 3: Predicting Cosmological Parameters from Merger Trees
- Data download: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/>
  - training set: `CS_tree_train.pt`
  - validation set: `CS_tree_val.pt`
  - test set: `CS_tree_test.pt`
- Nearest Neighbor based on KS-statistic: `models/tree_param/ks_neighbor.ipynb`
- DeepSets: `python train_tree_regression.py --model_type DeepSet`
- GNNs: `python train_tree_regression.py`

### Task 4: Merger Node Classification
- Data download: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/>
  - training set: `infilling_trees_25k_200_train.pt`
  - validation set: `infilling_trees_25k_200_val.pt`
  - test set: `infilling_trees_25k_200_test.pt`
- EPS cosmological baseline: See implementations in <https://github.com/nhuang37/tree_recon_EPS>
- K-nearest neighbors: `tree_infilling.ipynb`
- GNNs: `python train_tree_infilling.py`
