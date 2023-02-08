#!/bin/bash
for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 0 -1 0
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 2 -1 0
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 0 1 1
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 2 1 1
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 0 -1 1
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" 2 -1 1
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" -1 1 0
done

for i in `seq 1 10`; do
    python exec_modisco_perturb_models_covar_drop.py "${i}" -1 -1 0
done


