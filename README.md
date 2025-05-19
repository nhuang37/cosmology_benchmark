## CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

### Task 1: Predicting Cosmological Parameters from Point Clouds
- Linear Least Squares (LLS): `python models/cloud_param/simple_param.py`
- Graph Neural Networks (GNNs): See implementations in <https://github.com/Byeol-Haneul/TopoGal/tree/main>

### Task 2: Predicting Velocities from Point Positions
- LLS: `python models/cloud_velocity/simple_velocity.py`
- GNNs: `python train_velocity.py`

### Task 3: Predicting Cosmological Parameters from Merger Trees
- DeepSets: `python train_tree_regression.py --model_type DeepSet`
- GNNs: `python train_tree_regression.py`

### Task 4: Merger Node Classification
- K-nearest neighbors: `tree_infilling.ipynb`
- GNNs: `python train_tree_infilling.py`