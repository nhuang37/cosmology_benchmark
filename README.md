## CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

### Task 1: Predicting Cosmological Parameters from Point Clouds
- Data preparation: See details in [`models/cloud_param`](./models/cloud_param/)
- Linear Least Squares (LLS): `python models/cloud_param/simple_param.py`
- Graph Neural Networks (GNNs): See implementations in <https://github.com/Byeol-Haneul/TopoGal/tree/benchmark>

### Task 2: Predicting Velocities from Point Positions
- Data preparation: See details in [`models/cloud_velocity`](./models/cloud_velocity/)
- LLS: `python models/cloud_velocity/simple_velocity.py`
- GNNs: `python train_velocity.py`

### Task 3: Predicting Cosmological Parameters from Merger Trees
- Data download: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/>
  - training set: `CS_tree_train.pt`
  - validation set: `CS_tree_val.pt`
  - test set: `CS_tree_test.pt`
- DeepSets: `python train_tree_regression.py --model_type DeepSet`
- GNNs: `python train_tree_regression.py`

### Task 4: Merger Node Classification
- Data download: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/trees/>
  - training set: `infilling_trees_25k_200_train.pt`
  - validation set: `infilling_trees_25k_200_val.pt`
  - test set: `infilling_trees_25k_200_test.pt`
- K-nearest neighbors: `tree_infilling.ipynb`
- GNNs: `python train_tree_infilling.py`