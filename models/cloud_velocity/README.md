### Task 2: Predicting Velocities from Point Positions

- Linear Least Squares (LLS): 
  - Run linear least squares fit: `python models/cloud_velocity/simple_velocity.py --h5_path_train <DATA_TRAIN.h5> --h5_path_test <DATA_Test.h5> --K <K> --P <P>`
    - Default set-up: `CAMELS-SAM` with `--K 25 --P 3` 
    - For `Quijote` use `--K 15 --P 4` 
    - For `CAMELS` use `--K 10 --P 3` 
  - Evaluate pretrained models: 
    - Download pretrained weights to `OUTPUT_DIR`:  <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/<DATASET>/cloud_velocity/LLS/pretrained_models>
    - Run evaluation `python models/cloud_velocity/simple_velocity.py --h5_path_train <DATA_TRAIN.h5> --h5_path_test <DATA_Test.h5> --K <K> --P <P> --output_dir <OUTPUT_DIR> --eval_test`


- Graph Neural Networks (GNNs): 
  - Download precomputed graphs and features to `DATA_DIR`: 
    - `QUIJOTE_DATA_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/Quijote/cloud_velocity/GNN/features>
    - `CAMELS-SAM_DATA_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/galaxies/cloud_velocity/GNN/features>
    - `CAMELS-DATA_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS/cloud_velocity/GNN/features>
    
  - Train models: `python train_velocity.py --args.data_dir <DATA_DIR>`

  - Download pretrained models:
    - `QUIJOTE_OUTPUT_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/Quijote/cloud_velocity/GNN/pretrained_models>
    - `CAMELS-SAM_OUTPUT_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS-SAM/galaxies/cloud_velocity/GNN/pretrained_models>
    - `CAMELS_OUTPUT_DIR`: <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench/CAMELS/cloud_velocity/GNN/pretrained_models>

  - Evaluate pretrained models:
    - Run evaluation on Quijote `python train_velocity.py --data_dir <QUIJOTE_DATA_DIR> --processed_test_path Quijote_Rc=0.1_test.pt --output_dir <QUIJOTE_OUTPUT_DIR> --eval_test --train_all`
    - Run on CAMELS-SAM: `python train_velocity.py --data_dir <CAMELS-SAM_DATA_DIR>  --output_dir <CAMELS-SAM_OUTPUT_DIR> --processed_train_path CAMELS-SAM_Rc=0.1_train.pt --processed_val_path CAMELS-SAM_Rc=0.1_val.pt --processed_test_path CAMELS-SAM_Rc=0.1_test.pt --eval_test` 
    - Run evaluation on CAMELS `python train_velocity.py --data_dir <CAMELS-DATA_DIR> --processed_train_path CAMELS-TNG_Rc=0.1_train.pt --processed_val_path CAMELS-TNG_Rc=0.1_val.pt --processed_test_path CAMELS-TNG_Rc=0.1_test.pt --output_dir <CAMELS_OUTPUT_DIR> --eval_test`
