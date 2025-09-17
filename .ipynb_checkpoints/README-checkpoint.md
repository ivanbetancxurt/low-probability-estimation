# Low Probability Estimation
Public code to accompany ["Estimating the Probabilities of Rare Outputs in Language Models"](https://arxiv.org/abs/2410.13211). This repo contains:
- Code for QLD, MHIS, ITGIS, and GLD estimation methods (`lpe/methods.py`)
- Code to sample from the 8 input distributions (`lpe/utils/distributions`)
    - Note that the distributions "indent" and "ifelse" are the same as "colon" and "if", respectively, in the paper.
- Example code for generating and plotting (unfit) estimates versus ground truth (`example.ipynb`). Also available as a Colab notebook here: https://colab.research.google.com/github/alignment-research-center/low-probability-estimation/blob/main/example.ipynb

# Installation instructions
Clone this repo and install with `pip install -e PATH/TO/THIS/REPO`

# Public datafiles
Ground truth token frequencies for each distribution are available at `gs://arc-ml-public/lpe/ground-truth/MODEL_NAME/frequencies32-DIST_NAME.pt`, e.g., `gs://arc-ml-public/lpe/ground-truth/gelu-4l/frequencies32-hex.pt`
