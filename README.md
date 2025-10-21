## CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

[Website](https://cosmobench.streamlit.app) | [NeurIPS D&B Paper](https://arxiv.org/abs/2507.03707) | [Blogpost](https://medium.com/@sass1213teresa/cosmobench-a-cosmology-benchmark-for-geometric-deep-learning-2ebfeed8f324) 


### Summary of Datasets in **CosmoBench**

| **Dataset** | **Quijote** | **CAMELS-SAM** | **CAMELS** | **CS-Trees** |
|:-------------|:-------------:|:-------------:|:-------------:|:-------------:|
| **Modality** | Point Clouds | Point Clouds | Point Clouds | Directed Trees |
| **Box Size** | 1 000 cMpc / h | 100 cMpc / h | 25 cMpc / h | 100 cMpc / h |
| **Node Entity** | Halo | Galaxy | Galaxy | Halo |
| **Number of Graphs** | 32 752 | 1 000 | 1 000 | 24 996 |
| **Number of Nodes** | 5 000 | 5 000 | [588 – 4 511] | [121 – 37 865] |

### Data Exploration and Set-up
- All data can be downloaded from <https://users.flatironinstitute.org/~fvillaescusa/CosmoBench> (including precomputed graphs/features and pretrain models) or via [HuggingFace](https://huggingface.co/datasets/fvillaescusa/CosmoBench) 
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
- Cosmological baseline (2PCF): `python -m models.cloud_param.2PCF`
- Linear Least Squares (LLS): `python -m models.cloud_param.simple_param`
- Graph Neural Networks (GNNs): See implementations in <https://github.com/Byeol-Haneul/CosmoTopo/tree/benchmark>

### Task 2: Predicting Velocities from Point Positions
- Data preparation: See details in [`models/cloud_velocity`](./models/cloud_velocity/)
- Cosmological baseline (linear theory): `python -m models.cloud_velocity.linear_theory`
- LLS: `python -m models.cloud_velocity.simple_velocity`
- GNNs: `python train_velocity.py`

#### Task 2.1: Predicting Velocities from Positions in Redshift space
- The data and baselines in Task 2 operate on real-space positions, i.e., the ideal setting thanks to the controlled environment from cosmological simulations. However, in practice, the halo/galaxy positions are measured in redshift space, distored along the line of sight due to peculiar velocites.
- To construct redshift position, use the helper function `pos_redshift_space` in [`utils/get_redshift_pos.py`](./utils/get_redshift_pos.py)
- To train baselines above on redshift positions (with line-of-sight direction along z-axis), pass the argument `--eval_redshift`
  - Linear theory modification: `python -m models.cloud_velocity.linear_theory --eval_redshift --bias_LT`
  - LLS modification: `python -m models.cloud_velocity.simple_velocity --eval_redshift --shrink_factor 0.4`
  - For GNNs, we recompute the graphs `python -m utils.graph_util --redshift_flag`, then use the same training script   `python train_velocity.py`

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


### Use and Cite CosmoBench
If you use CosmoBench in your work, please cite our NeurIPS 2025 paper
```
@inproceedings{
huang2025cosmobench,
title={CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning},
author={Ningyuan Teresa Huang and Richard Stiskalek and Jun-Young Lee and Adrian E. Bayer and Charles Margossian and Christian Kragh Jespersen and Lucia A. Perez and Lawrence K. Saul and Francisco Villaescusa-Navarro},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=SSTilOugAu}
}
```
