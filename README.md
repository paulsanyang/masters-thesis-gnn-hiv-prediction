# Predicting Molecule Efficiency Against HIV Using Graph Neural Networks

**Master's Thesis — African Institute for Mathematical Sciences (AIMS)**
**Paul Sanyang** 

This repository contains selected portions of the code developed during my master's thesis and experimental analysis
conducted as part of my Master's thesis in **Mathematical Sciences - Machine Learning**.


This thesis studies graph neural networks for molecular activity classification on the [Open Graph Benchmark `ogbg-molhiv`](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) dataset (41,127 molecular graphs). Atoms are nodes and chemical bonds are edges.

## Research question

How does incorporating chemical bond information change graph convolution for molecular activity prediction?

## Implemented architectures

1. **Node-only GCN (baseline):** symmetric degree-normalized adjacency and atom features.
2. **Additive edge-aware GCN (thesis-inspired):** adds learned bond embeddings to messages from neighboring atoms.
3. **Edge-weighted GCN (new extension):** learns a scalar weight from bond embeddings and uses it to modify graph adjacency before normalization.

## Run the notebook

1. Install dependencies into the Jupyter kernel environment:

   ```bash
   python -m pip install torch torch-geometric ogb rdkit numpy pandas matplotlib networkx scikit-learn tqdm
   ```

2. Open `masters_thesis_aims_mol-hiv_paul.ipynb` and run cells in order. The dataset is downloaded automatically on first use.
3. Use `EPOCHS = 2` to test the pipeline. For a longer comparison, change to `EPOCHS = 120`.
4. The default `DEVICE_PREFERENCE = 'cpu'` works on most machines. Apple Silicon users can try `'mps'` if their installed PyG operations support it.
5. Select models based on **validation** ROC-AUC. Set `RUN_FINAL_TEST = True` only after model selection is finalized.

**Notes:** The notebook uses dense adjacency tensors, which can be memory-intensive. PyTorch 2.6+ may require trusted OGB processed-object loading; the notebook uses a temporary compatibility override for trusted official dataset files.

## Evaluation and visualizations

The notebook provides molecular graph plots, label imbalance, molecule size distributions, adjacency matrices, per-model loss curves, validation, and ROC-AUC curves.

## Research limitations

The dataset has a strong class imbalance. ROC-AUC is a ranking metric, not a claim of clinical effectiveness. For a rigorous follow-up, repeat each architecture across several random seeds and report mean and variability.

## References

Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. International Conference on Learning Representations (ICLR).
Paper: https://arxiv.org/abs/1609.02907

 Kipf, T. N., & Welling, M. (2016). Variational Graph Auto-Encoders. Bayesian Deep Learning Workshop, Neural Information Processing Systems (NIPS). 
 Paper: https://arxiv.org/abs/1611.07308

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. Advances in Neural Information Processing Systems, 33 (NeurIPS 2020).
link: https://proceedings.neurips.cc/paper_files/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html
