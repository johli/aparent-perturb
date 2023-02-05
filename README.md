A multi-task neural network for predicting perturbation responses to Alternative Polyadenylation usage levels
![APARENT-Perturb Logo](https://github.com/johli/aparent-perturb/blob/master/aparent_perturb_logo.png?raw=true)

# APARENT-Perturb
This repository contains the code for training APARENT-Perturb, a multi-task neural network that takes 3' UTR polyadenylation signal sequences as input (alongside APARENT2 baseline scores) to predict the impact of single-cell Perturb-seq perturbations on polyadenylation site usage.

Contact *jlinder2 (at) stanford.edu* for any questions about the model or data.

APARENT-Perturb does not require installation. Just clone or fork the [github repository](https://github.com/johli/aparent-perturb.git):
```sh
git clone https://github.com/johli/aparent-perturb.git
```

#### APARENT-Perturb requires the following packages to be installed
- Python >= 3.6
- Tensorflow == 1.13.1
- Keras == 2.2.4

## Data Availability
The processed data features (e.g. one-hot-coded sequence matrices and pseudo-bulked APA isoform proportions) are available at the link below. The link also houses prediction and interpretation results (e.g. ISM matrices) for all modelled perturbations.

[Processed Data Repository](https://drive.google.com/open?id=1LLJpMJUdrCTc9Bq0Cq6LBhAEBv2y9XER)<br/>

## Notebooks
The following notebook scripts contain scripts for processing the data, training models and applying interpretations methods to them.

[Notebook 0: Process Data Features (3' UTR only)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/data/process_native_data_features_utr3_polyadb_perturb.ipynb)<br/>
