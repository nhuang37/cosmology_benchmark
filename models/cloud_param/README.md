### Task 1: Predicting Cosmological Parameters from Point Clouds

- Linear Least Squares (LLS): 
  - Download precomputed features to `FEAT_PATH`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/<DATASET>/cloud_param/LLS/features>
    - Replace `DATASET` with `Quijote`, `CAMELS-SAM`, or `CAMELS`
  - Run linear least squares fit: `python models/cloud_param/simple_param.py --h5_path_train <DATA_TRAIN.h5> --h5_path_test <DATA_Test.h5>  --feature_dir FEAT_PATH`
    - Default set-up: `CAMELS-SAM`


- Graph Neural Networks (GNNs): See implementations in <https://github.com/Byeol-Haneul/TopoGal/tree/benchmark>
