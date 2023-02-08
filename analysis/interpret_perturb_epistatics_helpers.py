from __future__ import print_function

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

def contain_tf_gpu_mem_usage() :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

contain_tf_gpu_mem_usage()

import os
import pandas as pd

import numpy as np
import pickle

import matplotlib.pyplot as plt
#import seaborn as sns

from scipy.stats import ttest_ind
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize

from interpret_perturb_epistatics_helpers import *

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

#Visualization code

def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
                "UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
                "DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
                "(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
                "." : TextPath((-0.125, 0), "-", size=1, prop=fp),
                ")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange',#'orange', 
                    'A': 'green',#'red', 
                    'C': 'blue',#'blue', 
                    'T': 'red',#'darkgreen',
                    'UP': 'green', 
                    'DN': 'red',
                    '(': 'black',
                    '.': 'black', 
                    ')': 'black'}


    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None :
        chosen_color = color

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default") :

    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)
    
    ref_seq = ""
    for j in range(importance_scores.shape[1]) :
        argmax_nt = np.argmax(np.abs(importance_scores[:, j]))
        
        if argmax_nt == 0 :
            ref_seq += "A"
        elif argmax_nt == 1 :
            ref_seq += "C"
        elif argmax_nt == 2 :
            ref_seq += "G"
        elif argmax_nt == 3 :
            ref_seq += "T"

    ax = plt.gca()
    
    for i in range(0, len(ref_seq)) :
        mutability_score = np.sum(importance_scores[:, i])
        color = None
        dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)
    
    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.xlim((0, len(ref_seq)))
    
    #plt.axis('off')
    
    if plot_y_ticks :
        plt.yticks(fontsize=12)
    else :
        plt.yticks([], [])
    
    if y_min is not None and y_max is not None :
        plt.ylim(y_min, y_max)
    elif y_min is not None :
        plt.ylim(y_min)
    else :
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores))
        )
    
    plt.axhline(y=0., color='black', linestyle='-', linewidth=1)

    #for axis in fig.axes :
    #    axis.get_xaxis().set_visible(False)
    #    axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()

def plot_pwm_2(pwm, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default") :
    
    fig = plt.figure(figsize=figsize)

    ax = plt.gca()
    
    height_base = 0.
    logo_height = 1.0
    
    for j in range(0, pwm.shape[0]) :
        sort_index = np.argsort(pwm[j, :])

        for ii in range(0, 4) :
            i = sort_index[ii]

            nt_prob = pwm[j, i]# * conservation[j]

            nt = ''
            if i == 0 :
                nt = 'A'
            elif i == 1 :
                nt = 'C'
            elif i == 2 :
                nt = 'G'
            elif i == 3 :
                nt = 'T'

            color = None
            if ii == 0 :
                dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
            else :
                prev_prob = np.sum(pwm[j, sort_index[:ii]]) * logo_height # * conservation[j]
                dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)
    
    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.xlim((0, pwm.shape[0]))
    
    #plt.axis('off')
    
    if plot_y_ticks :
        plt.yticks(fontsize=12)
    else :
        plt.yticks([], [])
    
    if y_min is not None and y_max is not None :
        plt.ylim(y_min, y_max)
    elif y_min is not None :
        plt.ylim(y_min)
    else :
        plt.ylim(
            min(0., np.min(np.sum(pwm, axis=-1))) - 0.01 * np.max(np.abs(np.sum(pwm, axis=-1))),
            max(0., np.max(np.sum(pwm, axis=-1))) + 0.01 * np.max(np.abs(np.sum(pwm, axis=-1)))
        )
    
    print(np.min(np.sum(pwm, axis=-1)) - 0.1 * np.max(np.abs(np.sum(pwm, axis=-1))))
    print(np.max(np.sum(pwm, axis=-1)) + 0.1 * np.max(np.abs(np.sum(pwm, axis=-1))))
    
    plt.axhline(y=0., color='black', linestyle='-', linewidth=1)

    #for axis in fig.axes :
    #    axis.get_xaxis().set_visible(False)
    #    axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()

#Epistasis analysis code

bases = [0, 1, 2, 3]

def _predict(tissue_models, x) :
    
    y_hats = []
    for bootstrap_ix in range(len(tissue_models)) :
        y_hats.append(tissue_models[bootstrap_ix].predict(x=[x], batch_size=32)[..., None])
    
    return np.mean(np.concatenate(y_hats, axis=-1), axis=-1)

def _ablate_region(tissue_models, x, region, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix) :

    for [start_ix, end_ix] in region :

        for sample_ix in range(n_samples) :

            sampled_nts = np.random.choice(bases, size=(end_ix - start_ix)).tolist()

            x[sample_ix, 0, start_ix:end_ix, :] = 0.
            x[sample_ix, 0, np.arange(end_ix - start_ix) + start_ix, sampled_nts] = 1.
    
    s_abl = _predict(tissue_models, x)
    s_abl = s_abl[:, cell_type_2_ix, score_ix] - s_abl[:, cell_type_1_ix, score_ix]
    
    return x, s_abl

def _replace_region(tissue_models, x, region, region_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix) :

    #One-hot-encode replacements
    motifs = np.zeros((len(region_replacements), 1, len(region_replacements[0]), 4))

    #One-hot-encode motif 1
    for i in range(len(region_replacements)) :
        for j in range(len(region_replacements[i])) :
            if region_replacements[i][j] == 'A' :
                motifs[i, 0, j, 0] = 1.
            elif region_replacements[i][j] == 'C' :
                motifs[i, 0, j, 1] = 1.
            elif region_replacements[i][j] == 'G' :
                motifs[i, 0, j, 2] = 1.
            elif region_replacements[i][j] == 'T' :
                motifs[i, 0, j, 3] = 1.
    
    for [start_ix, end_ix] in region :

        for sample_ix in range(n_samples) :

            sampled_motif = motifs[np.random.choice(np.arange(len(region_replacements)).tolist()), ...]

            x[sample_ix, 0, start_ix:end_ix, :] = sampled_motif[0, :, :]
    
    s_abl = _predict(tissue_models, x)
    s_abl = s_abl[:, cell_type_2_ix, score_ix] - s_abl[:, cell_type_1_ix, score_ix]
    
    return x, s_abl

def _analyze_epistatics_old(i=0, n_samples=1024, cell_type_1_ix=0, cell_type_2_ix=6, score_ix=2, region_1=[[0, 10]], region_2=[[10, 20]], save_figs=False, fig_name='default') :
    
    #Predict reference
    s_ref = _predict(tissue_models, flat_x[i:i+1])[0]
    s_ref = s_ref[cell_type_2_ix, score_ix] - s_ref[cell_type_1_ix, score_ix]

    #Ablate each region independently and both regions concurrently
    x_abl_1 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
    x_abl_2 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
    x_abl_both = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))

    x_abl_1, s_abl_1 = _ablate_region(tissue_models, x_abl_1, region_1, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
    x_abl_2, s_abl_2 = _ablate_region(tissue_models, x_abl_2, region_2, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)

    for [start_ix, end_ix] in region_1 :
        x_abl_both[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]

    for [start_ix, end_ix] in region_2 :
        x_abl_both[..., start_ix:end_ix, :] = x_abl_2[..., start_ix:end_ix, :]

    s_abl_both = _predict(tissue_models, x_abl_both)
    s_abl_both = s_abl_both[:, cell_type_2_ix, score_ix] - s_abl_both[:, cell_type_1_ix, score_ix]

    epi_score = (s_abl_both - s_ref) - ((s_abl_2 - s_ref) + (s_abl_1 - s_ref))

    print("s_abl_both - s_ref (median) = " + str(round(np.median(s_abl_both - s_ref), 4)))
    print("s_abl_1 - s_ref (median) = " + str(round(np.median(s_abl_1 - s_ref), 4)))
    print("s_abl_2 - s_ref (median) = " + str(round(np.median(s_abl_2 - s_ref), 4)))

    f = plt.figure(figsize=(4, 4))

    plt.hist(x=np.exp(-epi_score), bins=25, color='deepskyblue', linewidth=2, edgecolor='black')

    plt.axvline(x=np.exp(np.median(-epi_score)), color='red', linewidth=3, linestyle='--')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("Epistasis (odds ratio)", fontsize=12)
    plt.ylabel("# Samples", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_epistasis_score_example_" + str(i) + "_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_epistasis_score_example_" + str(i) + "_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".eps")

    plt.show()

def _analyze_epistatics(i=0, n_samples=1024, cell_type_1_ix=0, cell_type_2_ix=6, score_ix=2, region_1=[[0, 10]], region_2=[[10, 20]], region_1_replacements=None, region_2_replacements=None, epi_sign=1., save_figs=False, fig_name='default') :
    
    #Predict reference
    s_ref = _predict(tissue_models, flat_x[i:i+1])[0]
    s_ref = s_ref[cell_type_2_ix, score_ix] - s_ref[cell_type_1_ix, score_ix]

    #Ablate each region independently and both regions concurrently
    x_abl_1 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
    x_abl_2 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
    x_abl_both = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))

    if region_1_replacements is None :
        x_abl_1, s_abl_1 = _ablate_region(tissue_models, x_abl_1, region_1, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
    else :
        x_abl_1, s_abl_1 = _replace_region(tissue_models, x_abl_1, region_1, region_1_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
    if region_2_replacements is None :
        x_abl_2, s_abl_2 = _ablate_region(tissue_models, x_abl_2, region_2, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
    else :
        x_abl_2, s_abl_2 = _replace_region(tissue_models, x_abl_2, region_2, region_2_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
    
    for [start_ix, end_ix] in region_1 :
        x_abl_both[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]

    for [start_ix, end_ix] in region_2 :
        x_abl_both[..., start_ix:end_ix, :] = x_abl_2[..., start_ix:end_ix, :]

    s_abl_both = _predict(tissue_models, x_abl_both)
    s_abl_both = s_abl_both[:, cell_type_2_ix, score_ix] - s_abl_both[:, cell_type_1_ix, score_ix]

    epi_score = (s_abl_both - s_ref) - ((s_abl_2 - s_ref) + (s_abl_1 - s_ref))

    print("s_abl_both - s_ref (median) = " + str(round(np.median(s_abl_both - s_ref), 4)))
    print("s_abl_1 - s_ref (median) = " + str(round(np.median(s_abl_1 - s_ref), 4)))
    print("s_abl_2 - s_ref (median) = " + str(round(np.median(s_abl_2 - s_ref), 4)))

    f = plt.figure(figsize=(4, 4))

    plt.hist(x=np.exp(epi_sign * -epi_score), bins=25, color='deepskyblue', linewidth=2, edgecolor='black')

    plt.axvline(x=np.exp(np.median(epi_sign * -epi_score)), color='red', linewidth=3, linestyle='--')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("Epistasis (odds ratio)", fontsize=12)
    plt.ylabel("# Samples", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_epistasis_score_example_" + str(i) + "_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_epistasis_score_example_" + str(i) + "_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".eps")

    plt.show()

def _analyze_epistatics_many(subset_index, pwm_match_mask, pwm_match_index, pwm_match_poses, cell_type_1_ix=0, cell_type_2_ix=6, score_ix=2, n_samples=32, _valid_null_pos_func=lambda a, b, c: np.arange(205).tolist(), hypo_str="Dual TGTA", save_figs=False, fig_name='default') :
    
    pwm_match_dict = {}
    for match_ix, match_pos in zip(pwm_match_index, pwm_match_poses) :
        if match_ix not in pwm_match_dict :
            pwm_match_dict[match_ix] = []

        pwm_match_dict[match_ix].append(match_pos)

    for match_ix in pwm_match_dict :
        pwm_match_dict[match_ix] = np.sort(np.unique(np.array(pwm_match_dict[match_ix], dtype=np.int32))).tolist()

    epi_scores = []
    epi_scores_null = []

    for ii, i in enumerate(subset_index) :

        if ii % 100 == 0 :
            print("Processing example " + str(ii) + "...")

        start_1 = pwm_match_dict[i][0]
        start_2 = pwm_match_dict[i][1]

        if start_1 > start_2 - pattern_width :
            continue

        valid_null_poses = _valid_null_pos_func(start_1, start_2, pattern_width)

        start_2_null = np.random.choice(valid_null_poses)

        region_1 = [[start_1, start_1+pattern_width]]
        region_2 = [[start_2, start_2+pattern_width]]
        region_2_null = [[start_2_null, start_2_null+pattern_width]]

        #Predict reference
        s_ref = _predict(tissue_models, flat_x[i:i+1])[0]
        s_ref = s_ref[cell_type_2_ix, score_ix] - s_ref[cell_type_1_ix, score_ix]

        #Ablate each region independently and both regions concurrently
        x_abl_1 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_2 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_2_null = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_both = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_both_null = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))

        x_abl_1, s_abl_1 = _ablate_region(tissue_models, x_abl_1, region_1, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        x_abl_2, s_abl_2 = _ablate_region(tissue_models, x_abl_2, region_2, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        x_abl_2_null, s_abl_2_null = _ablate_region(tissue_models, x_abl_2_null, region_2_null, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)

        for [start_ix, end_ix] in region_1 :
            x_abl_both[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]
        for [start_ix, end_ix] in region_2 :
            x_abl_both[..., start_ix:end_ix, :] = x_abl_2[..., start_ix:end_ix, :]

        for [start_ix, end_ix] in region_1 :
            x_abl_both_null[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]
        for [start_ix, end_ix] in region_2_null :
            x_abl_both_null[..., start_ix:end_ix, :] = x_abl_2_null[..., start_ix:end_ix, :]

        s_abl_both = _predict(tissue_models, x_abl_both)
        s_abl_both = s_abl_both[:, cell_type_2_ix, score_ix] - s_abl_both[:, cell_type_1_ix, score_ix]

        s_abl_both_null = _predict(tissue_models, x_abl_both_null)
        s_abl_both_null = s_abl_both_null[:, cell_type_2_ix, score_ix] - s_abl_both_null[:, cell_type_1_ix, score_ix]

        epi_score = (s_abl_both - s_ref) - ((s_abl_2 - s_ref) + (s_abl_1 - s_ref))
        epi_score_null = (s_abl_both_null - s_ref) - ((s_abl_2_null - s_ref) + (s_abl_1 - s_ref))

        epi_scores.append(np.mean(epi_score))
        epi_scores_null.append(np.mean(epi_score_null))

    epi_scores = np.array(epi_scores)
    epi_scores_null = np.array(epi_scores_null)

    print("epi_scores.shape = " + str(epi_scores.shape))
    print("epi_scores_null.shape = " + str(epi_scores_null.shape))
    
    import seaborn as sns
    from scipy.stats import ranksums

    s_val, p_val = ranksums(epi_scores, epi_scores_null, alternative='two-sided')

    print("wilcoxon p = " + str(p_val))

    f = plt.figure(figsize=(6, 4))

    sns.stripplot(data=[np.exp(-epi_scores), np.exp(-epi_scores_null)], alpha=0.9, jitter=0.25, palette=['red', 'green'])
    sns.boxplot(data=[np.exp(-epi_scores), np.exp(-epi_scores_null)], linewidth=2, fliersize=0., palette=['lightcoral', 'lightgreen'])

    plt.xticks([0, 1], [hypo_str, "Matched Control"], fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylabel("Epistasis (odds ratio)", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_epistasis_score_many_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_epistasis_score_many_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".eps")

    plt.show()

def _analyze_epistatics_many_heterotypic(subset_index, pwm_match_mask, pwm_match_mask_other, pwm_match_index, pwm_match_index_other, pwm_match_poses, pwm_match_poses_other, region_1_replacements=None, region_2_replacements=None, cell_type_1_ix=0, cell_type_2_ix=6, score_ix=2, n_samples=32, _valid_null_pos_func=lambda a, b, c: np.arange(205).tolist(), start_1_pos_constraint=70, start_2_pos_constraint=76, epi_sign=1., hypo_str="Dual TGTA", save_figs=False, fig_name='default') :

    pwm_match_dict = {}
    for match_ix, match_pos in zip(pwm_match_index, pwm_match_poses) :
        if match_ix not in pwm_match_dict :
            pwm_match_dict[match_ix] = []

        pwm_match_dict[match_ix].append(match_pos)

    for match_ix in pwm_match_dict :
        pwm_match_dict[match_ix] = np.sort(np.unique(np.array(pwm_match_dict[match_ix], dtype=np.int32))).tolist()

    pwm_match_dict_other = {}
    for match_ix, match_pos in zip(pwm_match_index_other, pwm_match_poses_other) :
        if match_ix not in pwm_match_dict_other :
            pwm_match_dict_other[match_ix] = []

        pwm_match_dict_other[match_ix].append(match_pos)

    for match_ix in pwm_match_dict_other :
        pwm_match_dict_other[match_ix] = np.sort(np.unique(np.array(pwm_match_dict_other[match_ix], dtype=np.int32))).tolist()

    epi_scores = []
    epi_scores_null = []

    for ii, i in enumerate(subset_index) :

        if ii % 100 == 0 :
            print("Processing example " + str(ii) + "...")

        valid_null_poses = _valid_null_pos_func(i, -1, -1, pattern_width, pwm_match_mask, pwm_match_mask_other)
        
        if start_1_pos_constraint is not None and start_2_pos_constraint is not None :
            region_1 = [[start_1, start_1+pattern_width] for start_1 in pwm_match_dict[i] if start_1 < start_1_pos_constraint]
            region_2 = [[start_2, start_2+pattern_width] for start_2 in pwm_match_dict_other[i] if start_2 >= start_2_pos_constraint]
        else :
            region_1 = [[start_1, start_1+pattern_width] for start_1 in pwm_match_dict[i]]
            region_2 = [[start_2, start_2+pattern_width] for start_2 in pwm_match_dict_other[i]]

        start_2_nulls = np.random.choice(valid_null_poses, size=(len(region_2),)).tolist()

        region_2_null = [[start_2_null, start_2_null+pattern_width] for start_2_null in start_2_nulls]

        if len(region_1) <= 0 or len(region_2) <= 0 :
            continue

        #Predict reference
        s_ref = _predict(tissue_models, flat_x[i:i+1])[0]
        s_ref = s_ref[cell_type_2_ix, score_ix] - s_ref[cell_type_1_ix, score_ix]

        #Ablate each region independently and both regions concurrently
        x_abl_1 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_2 = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_2_null = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_both = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))
        x_abl_both_null = np.tile(np.copy(flat_x[i:i+1]), (n_samples, 1, 1, 1))

        if region_1_replacements is None :
            x_abl_1, s_abl_1 = _ablate_region(tissue_models, x_abl_1, region_1, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        else :
            x_abl_1, s_abl_1 = _replace_region(tissue_models, x_abl_1, region_1, region_1_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        if region_2_replacements is None :
            x_abl_2, s_abl_2 = _ablate_region(tissue_models, x_abl_2, region_2, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        else :
            x_abl_2, s_abl_2 = _replace_region(tissue_models, x_abl_2, region_2, region_2_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        if region_2_replacements is None :
            x_abl_2_null, s_abl_2_null = _ablate_region(tissue_models, x_abl_2_null, region_2_null, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)
        else :
            x_abl_2_null, s_abl_2_null = _replace_region(tissue_models, x_abl_2_null, region_2_null, region_2_replacements, n_samples, cell_type_2_ix, cell_type_1_ix, score_ix)

        for [start_ix, end_ix] in region_1 :
            x_abl_both[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]
        for [start_ix, end_ix] in region_2 :
            x_abl_both[..., start_ix:end_ix, :] = x_abl_2[..., start_ix:end_ix, :]

        for [start_ix, end_ix] in region_1 :
            x_abl_both_null[..., start_ix:end_ix, :] = x_abl_1[..., start_ix:end_ix, :]
        for [start_ix, end_ix] in region_2_null :
            x_abl_both_null[..., start_ix:end_ix, :] = x_abl_2_null[..., start_ix:end_ix, :]

        s_abl_both = _predict(tissue_models, x_abl_both)
        s_abl_both = s_abl_both[:, cell_type_2_ix, score_ix] - s_abl_both[:, cell_type_1_ix, score_ix]

        s_abl_both_null = _predict(tissue_models, x_abl_both_null)
        s_abl_both_null = s_abl_both_null[:, cell_type_2_ix, score_ix] - s_abl_both_null[:, cell_type_1_ix, score_ix]

        epi_score = (s_abl_both - s_ref) - ((s_abl_2 - s_ref) + (s_abl_1 - s_ref))
        epi_score_null = (s_abl_both_null - s_ref) - ((s_abl_2_null - s_ref) + (s_abl_1 - s_ref))

        epi_scores.append(np.mean(epi_score))
        epi_scores_null.append(np.mean(epi_score_null))

    epi_scores = np.array(epi_scores)
    epi_scores_null = np.array(epi_scores_null)

    print("epi_scores.shape = " + str(epi_scores.shape))
    print("epi_scores_null.shape = " + str(epi_scores_null.shape))
    
    import seaborn as sns
    from scipy.stats import ranksums

    s_val, p_val = ranksums(epi_sign * epi_scores, epi_sign*epi_scores_null, alternative='two-sided')

    print("wilcoxon p = " + str(p_val))

    f = plt.figure(figsize=(6, 4))

    sns.stripplot(data=[np.exp(epi_sign * -epi_scores), np.exp(epi_sign * -epi_scores_null)], alpha=0.9, jitter=0.25, palette=['red', 'green'])
    sns.boxplot(data=[np.exp(epi_sign * -epi_scores), np.exp(epi_sign * -epi_scores_null)], linewidth=2, fliersize=0., palette=['lightcoral', 'lightgreen'])

    plt.xticks([0, 1], [hypo_str, "Matched Control"], fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylabel("Epistasis (odds ratio)", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_epistasis_score_many_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_epistasis_score_many_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".eps")

    plt.show()

#Code for multiple insertional motif simulation experiments

def _run_insertional_motif_simulation(sim_seqs, sim_motifs, sim_pos_funcs, sim_names, cell_type_1_ix=0, cell_type_2_ix=6, score_ix=2, load_from_cache=False, cache_name=None, hypo_names=['default'], sim_indexes=[[0]], sim_color_maps=['red'], x_ranges=[[0, 50]], plot_areas=[True], fig_names=['default'], plot_label_outside=False, epi_sign=1., save_figs=False) :
    
    #Run multiple insertional simulation experiments
    
    if not load_from_cache :
        sim_epi_scores = []

        for sim_ix in range(len(sim_names)) :

            sim_name = sim_names[sim_ix]
            sim_motif_pair = sim_motifs[sim_ix]
            sim_pos_func_pair = sim_pos_funcs[sim_ix]
            sim_seq_strs = sim_seqs[sim_ix]

            print("Running simulation '" + str(sim_name) + "' [sim_ix = " + str(sim_ix) + "]...")

            motif_1_seq, motif_2_seq = sim_motif_pair
            pos_1_func, pos_2_func = sim_pos_func_pair

            motif_1_len = len(motif_1_seq)
            motif_2_len = len(motif_2_seq)

            poses_1 = pos_1_func(motif_1_len)
            poses_2 = pos_2_func(motif_2_len)

            motif_1 = np.zeros((1, 1, motif_1_len, 4))
            motif_2 = np.zeros((1, 1, motif_2_len, 4))

            #One-hot-encode motif 1
            for j in range(len(motif_1_seq)) :
                if motif_1_seq[j] == 'A' :
                    motif_1[0, 0, j, 0] = 1.
                elif motif_1_seq[j] == 'C' :
                    motif_1[0, 0, j, 1] = 1.
                elif motif_1_seq[j] == 'G' :
                    motif_1[0, 0, j, 2] = 1.
                elif motif_1_seq[j] == 'T' :
                    motif_1[0, 0, j, 3] = 1.

            #One-hot-encode motif 2
            for j in range(len(motif_2_seq)) :
                if motif_2_seq[j] == 'A' :
                    motif_2[0, 0, j, 0] = 1.
                elif motif_2_seq[j] == 'C' :
                    motif_2[0, 0, j, 1] = 1.
                elif motif_2_seq[j] == 'G' :
                    motif_2[0, 0, j, 2] = 1.
                elif motif_2_seq[j] == 'T' :
                    motif_2[0, 0, j, 3] = 1.

            epi_scores = np.zeros((n_wt, len(poses_1), len(poses_2)))

            for wt_ix, wt_seq in enumerate(sim_seq_strs) :

                print(" - Processing wt_ix = " + str(wt_ix) + "...")

                #One-hot-encode simulation background sequence
                flat_x_wt = np.zeros((1, 1, 205, 4))
                for j in range(len(wt_seq)) :
                    if wt_seq[j] == 'A' :
                        flat_x_wt[0, 0, j, 0] = 1.
                    elif wt_seq[j] == 'C' :
                        flat_x_wt[0, 0, j, 1] = 1.
                    elif wt_seq[j] == 'G' :
                        flat_x_wt[0, 0, j, 2] = 1.
                    elif wt_seq[j] == 'T' :
                        flat_x_wt[0, 0, j, 3] = 1.

                s_wt = _predict(tissue_models, flat_x_wt)
                s_wt = s_wt[:, cell_type_2_ix, score_ix] - s_wt[:, cell_type_1_ix, score_ix]

                for pos_1_ix, pos_1 in enumerate(poses_1) :

                    flat_x_insert_1 = np.zeros((len(poses_2), 1, flat_x.shape[2], 4))
                    flat_x_insert_2 = np.zeros((len(poses_2), 1, flat_x.shape[2], 4))
                    flat_x_insert_both = np.zeros((len(poses_2), 1, flat_x.shape[2], 4))

                    flat_x_insert_1[:, ...] = np.tile(flat_x_wt[:, ...], (len(poses_2), 1, 1, 1))
                    flat_x_insert_2[:, ...] = np.tile(flat_x_wt[:, ...], (len(poses_2), 1, 1, 1))
                    flat_x_insert_both[:, ...] = np.tile(flat_x_wt[:, ...], (len(poses_2), 1, 1, 1))

                    flat_x_insert_1[:, 0, pos_1:pos_1+motif_1.shape[2], :] = motif_1
                    flat_x_insert_both[:, 0, pos_1:pos_1+motif_1.shape[2], :] = motif_1

                    for pos_2_ix, pos_2 in enumerate(poses_2) :

                        flat_x_insert_2[pos_2_ix, 0, pos_2:pos_2+motif_2.shape[2], :] = motif_2
                        flat_x_insert_both[pos_2_ix, 0, pos_2:pos_2+motif_2.shape[2], :] = motif_2

                    s_insert_1 = _predict(tissue_models, flat_x_insert_1)
                    s_insert_1 = s_insert_1[:, cell_type_2_ix, score_ix] - s_insert_1[:, cell_type_1_ix, score_ix]

                    s_insert_2 = _predict(tissue_models, flat_x_insert_2)
                    s_insert_2 = s_insert_2[:, cell_type_2_ix, score_ix] - s_insert_2[:, cell_type_1_ix, score_ix]

                    s_insert_both = _predict(tissue_models, flat_x_insert_both)
                    s_insert_both = s_insert_both[:, cell_type_2_ix, score_ix] - s_insert_both[:, cell_type_1_ix, score_ix]

                    epi_scores[wt_ix, pos_1_ix, :] = epi_sign * -1. * ((s_insert_both - s_wt) - ((s_insert_1 - s_wt) + (s_insert_2 - s_wt)))

            sim_epi_scores.append(epi_scores)

        #Aggregate epistatics by linear motif distance
        sim_epi_scores_by_distance = []

        for sim_ix in range(len(sim_names)) :

            motif_1_seq, motif_2_seq = sim_motifs[sim_ix]
            pos_1_func, pos_2_func = sim_pos_funcs[sim_ix]

            epi_scores = sim_epi_scores[sim_ix]

            motif_1_len = len(motif_1_seq)
            motif_2_len = len(motif_2_seq)

            poses_1 = pos_1_func(motif_1_len)
            poses_2 = pos_2_func(motif_2_len)

            epi_scores_by_distance = np.zeros((n_wt, 70-motif_1_len-motif_2_len))
            n_by_distance = np.zeros((n_wt, 70-motif_1_len-motif_2_len))

            for wt_ix in range(n_wt) :
                for pos_1_ix, pos_1 in enumerate(poses_1) :
                    for pos_2_ix, pos_2 in enumerate(poses_2) :

                        dist = pos_2 - pos_1 - motif_1_len

                        if dist < 0 :
                            continue

                        epi_scores_by_distance[wt_ix, dist] += epi_scores[wt_ix, pos_1_ix, pos_2_ix]
                        n_by_distance[wt_ix, dist] += 1.

            epi_scores_by_distance /= np.maximum(n_by_distance, 1.)

            sim_epi_scores_by_distance.append(epi_scores_by_distance)

    #Cache simulation results
    import pickle
    
    if not load_from_cache and cache_name is not None :
        pickle.dump({
            "sim_names" : sim_names,
            "sim_motifs" : sim_motifs,
            "sim_seqs" : sim_seqs,
            "sim_epi_scores" : sim_epi_scores,
            "sim_epi_scores_by_distance" : sim_epi_scores_by_distance,
        }, open(cache_name + ".pickle", 'wb'))
    elif load_from_cache and cache_name is not None :
        #Load simulation results
        cache_dict = pickle.load(open(cache_name + ".pickle", 'rb'))

        sim_names = cache_dict['sim_names']
        sim_motifs = cache_dict['sim_motifs']
        sim_seqs = cache_dict['sim_seqs']
        sim_epi_scores = cache_dict['sim_epi_scores']
        sim_epi_scores_by_distance = cache_dict['sim_epi_scores_by_distance']
    
    #Plot simulatated motif distance epistasis profiles
    for hypo_name, sim_index, sim_colors, x_range, plot_area, fig_name in zip(hypo_names, sim_indexes, sim_color_maps, x_ranges, plot_areas, fig_names) :
        
        min_x = x_range[0]
        max_x = x_range[1]

        x_vals = np.arange(max_x - min_x) + min_x

        f = None
        if not plot_label_outside :
            f = plt.figure(figsize=(6, 4))
        else :
            f = plt.figure(figsize=(9.5, 4))

        for sim_ix, sim_color in zip(sim_index, sim_colors) :

            epi_scores_by_distance = sim_epi_scores_by_distance[sim_ix]

            sim_name = sim_names[sim_ix]

            y_vals_low = np.exp(np.quantile(epi_scores_by_distance[:, min_x:max_x], q=0.1, axis=0))
            y_vals = np.exp(np.median(epi_scores_by_distance[:, min_x:max_x], axis=0))
            y_vals_high = np.exp(np.quantile(epi_scores_by_distance[:, min_x:max_x], q=0.9, axis=0))

            if plot_area :
                plt.plot(x_vals, y_vals_low, color=sim_color, alpha=0.5, linewidth=2, linestyle='--')
                plt.plot(x_vals, y_vals_high, color=sim_color, alpha=0.5, linewidth=2, linestyle='--')
                plt.fill_between(x_vals, y_vals_low, y_vals_high, color=sim_color, alpha=0.25)

            plt.plot(x_vals, y_vals, color=sim_color, linewidth=2, linestyle='-')
            plt.scatter(x_vals, y_vals, s=25, color=sim_color, label=sim_name)

        plt.axhline(y=1., linewidth=3, linestyle='--', color='black', alpha=0.75)

        plt.xlim(min_x-0.1, max_x-1+0.1)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.xlabel("Motif Distance (bp)", fontsize=12)
        plt.ylabel("Epistasis (odds ratio)", fontsize=12)

        plt.title(hypo_name, fontsize=12)

        if not plot_label_outside :
            plt.legend(fontsize=12)
        else :
            plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if save_figs :
            plt.savefig(fig_name + "_simulation_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".png", transparent=True, dpi=600)
            plt.savefig(fig_name + "_simulation_cell_type_2_ix_" + str(cell_type_2_ix) + "_score_ix_" + str(score_ix) + ".eps")

        plt.show()

#Code for feature regression and coefficient visualization

def _perform_regression(x_feat, motifs, nth_order, keep_index, cell_type_1_ix=0, cell_type_2_ix=6, n_test=500) :
    
    #Construct train/test feature matrices for regression
    cell_type_1_ix_global = cell_type_dict[subset_cell_types[cell_type_1_ix]]
    cell_type_2_ix_global = cell_type_dict[subset_cell_types[cell_type_2_ix]]

    #Filter data
    x_feat_kept = x_feat[keep_index, ...]

    m_kept = m[keep_index, ...]
    dist_mask_kept = dist_mask[keep_index, ...]

    y_dist_kept = y_dist[keep_index, ...]

    #Collapse features over PASs
    x_feat_prox_kept = np.sum((m_kept * (1. - dist_mask_kept))[..., None] * x_feat_kept, axis=1)
    x_feat_dist_kept = np.sum((m_kept * dist_mask_kept)[..., None] * x_feat_kept, axis=1)

    x_feat_kept = np.concatenate([x_feat_prox_kept, x_feat_dist_kept], axis=-1)

    #Estimate perturbation log odds ratio
    y_lor_kept = np.log(y_dist_kept[:, cell_type_1_ix_global] / (1. - y_dist_kept[:, cell_type_1_ix_global])) - np.log(y_dist_kept[:, cell_type_2_ix_global] / (1. - y_dist_kept[:, cell_type_2_ix_global]))

    print("y_lor_kept.shape = " + str(y_lor_kept.shape))

    #Get train/test splits
    x_feat_prox_train = x_feat_prox_kept[:-n_test]
    x_feat_dist_train = x_feat_dist_kept[:-n_test]

    x_feat_train = np.concatenate([x_feat_prox_train, x_feat_dist_train], axis=-1)

    y_lor_train = y_lor_kept[:-n_test]

    x_feat_prox_test = x_feat_prox_kept[-n_test:]
    x_feat_dist_test = x_feat_dist_kept[-n_test:]

    x_feat_test = np.concatenate([x_feat_prox_test, x_feat_dist_test], axis=-1)

    y_lor_test = y_lor_kept[-n_test:]

    print("x_feat_train.shape = " + str(x_feat_train.shape))
    print("x_feat_test.shape = " + str(x_feat_test.shape))

    print("y_lor_train.shape = " + str(y_lor_train.shape))
    print("y_lor_test.shape = " + str(y_lor_test.shape))
    print("")
    
    #Print total feature occurrence count
    print(np.sum(x_feat_prox_kept, axis=0))
    print(np.sum(x_feat_dist_kept, axis=0))

    #Run regression (bootstrapped and cross-validated versions)
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import KFold

    #Train single model on train / test split
    lr_model = LinearRegression().fit(x_feat_train, y_lor_train)

    y_lor_hat_train = lr_model.predict(x_feat_train)
    y_lor_hat_test = lr_model.predict(x_feat_test)

    #Train cross-validated models
    n_folds = 50

    kf = KFold(n_splits=n_folds)

    y_lor_hat_kept = np.zeros(y_lor_kept.shape)

    for fold_ix, [train_index, test_index] in enumerate(kf.split(x_feat_kept)) :

        x_feat_train_fold = x_feat_kept[train_index]
        x_feat_test_fold = x_feat_kept[test_index]

        y_lor_train_fold = y_lor_kept[train_index]

        y_lor_hat_test_fold = LinearRegression().fit(x_feat_train_fold, y_lor_train_fold).predict(x_feat_test_fold)
        y_lor_hat_kept[test_index] = y_lor_hat_test_fold[:]

    #Train bootstraps
    n_bootstrap_samples = 1000

    coefs = []

    y_lor_hat_trains = []
    y_lor_hat_tests = []

    for bootstrap_ix in range(n_bootstrap_samples) :

        bootstrap_index = np.random.choice(np.arange(x_feat_train.shape[0]), size=(x_feat_train.shape[0],), replace=True)

        if bootstrap_ix % 500 == 0 :
            print("Bootstrap = " + str(bootstrap_ix))

        lr_model_bootstrapped = LinearRegression().fit(x_feat_train[bootstrap_index, ...], y_lor_train[bootstrap_index, ...])
        #lr_model = Ridge(alpha=0.1).fit(x_feat_train, y_lor_train)

        coefs.append(np.copy(lr_model_bootstrapped.coef_)[None, :])

    coefs = np.concatenate(coefs, axis=0)
    
    #Performance on Train / Test splits and CV performance
    from scipy.stats import spearmanr, pearsonr

    r_train = spearmanr(y_lor_hat_train, y_lor_train)[0]
    r_test = spearmanr(y_lor_hat_test, y_lor_test)[0]
    r_cv = spearmanr(y_lor_hat_kept, y_lor_kept)[0]

    print("(train) r = " + str(round(r_train, 3)))
    print(" (test) r = " + str(round(r_test, 3)))
    print("   (cv) r = " + str(round(r_cv, 3)))
    
    return coefs
