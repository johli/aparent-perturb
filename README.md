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
The following notebook scripts contain scripts for processing the data, training models and applying interpretation methods to them.

[Notebook 0a: Process Data](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/data/process_native_data_features_polyadb_perturb.ipynb)<br/>
[Notebook 0b: Process Data (3' UTR only)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/data/process_native_data_features_utr3_polyadb_perturb.ipynb)<br/>
[Notebook 0c: Predict Non-targeting Controls (with APARENT2)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/predict_perturb_data_aparent_resnet_utr3.ipynb)<br/>
<br/>
[Notebook 1a: Train APARENT-Perturb](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/model/train_perturb_apa_model_resnet_shared_regr_w_covar_drop.ipynb)<br/>
[Notebook 1b: Cross-Validate APARENT-Perturb](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/model/train_perturb_apa_model_resnet_shared_regr_w_covar_drop_crossval_3_attempts.ipynb)<br/>
<br/>
[Notebook 2a: Interpret APARENT-Perturb (Windowed ISM)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/interpret_perturb_apa_model_covar_shuffled_window_ism.ipynb)<br/>
[Notebook 2b: Interpret APARENT-Perturb (Epistasis)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/interpret_perturb_apa_model_covar_epistatics.ipynb)<br/>
<br/>
[Notebook 3a: Predict PAF Perturbation (Intronic)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/intronic_pa/predict_perturb_data_aparent_resnet_PAF.ipynb)<br/>
[Notebook 3b: Predict PAF Perturbation (3' UTR; Control)](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/intronic_pa/predict_perturb_data_aparent_resnet_PAF_utr3.ipynb)<br/>
[Notebook 3c: Intronic Site Strength vs. Distance](https://nbviewer.jupyter.org/github/johli/aparent-perturb/blob/master/analysis/intronic_pa/predict_polyadb_data_aparent_resnet_intron.ipynb)<br/>
