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

import h5py

import matplotlib

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

#Visualization code

def ic_scale(pwm,background):
    per_position_ic = util.compute_per_position_ic(
                       ppm=pwm, background=background, pseudocount=0.001)
    return pwm*(per_position_ic[:,None])

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ylabel=""):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                 figsize=(20,2),
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,**kwargs)
    plt.show()


def plot_score_track_given_ax(arr, ax, threshold=None, **kwargs):
    ax.plot(np.arange(len(arr)), arr, **kwargs)
    if (threshold is not None):
        ax.plot([0, len(arr)-1], [threshold, threshold])
    ax.set_xlim(0,len(arr)-1)

def plot_score_track(arr, threshold=None, figsize=(20,2), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_score_track_given_ax(arr, threshold=threshold, ax=ax, **kwargs) 
    plt.show()

def plot_pwm(weights, figsize=(16, 2), plot_y_ticks=True, y_min=None, y_max=None, save_figs=False, fig_name="default") :
    colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
    
    plot_funcs = {0: plot_a, 1: plot_c, 
                  2: plot_g, 3: plot_t}

    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(111) 
    
    plot_weights_given_ax(ax=ax, array=weights, 
                                       height_padding_factor=0.2,
                                       length_padding=1.0, 
                                       subticks_frequency=1.0, 
                                       colors=colors, plot_funcs=plot_funcs, 
                                       highlight={}, ylabel="")

    plt.sca(ax)
    
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    if plot_y_ticks :
        plt.yticks(fontsize=12)
    else :
        plt.yticks([], [])
    
    if y_min is not None and y_max is not None :
        plt.ylim(y_min, y_max)
    elif y_min is not None :
        plt.ylim(y_min)
    
    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + ".eps")
    
    plt.show()

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

#Enumerate and print interpretations

def _print_interpretation(cell_type_1_ix=0, cell_type_2_ix=1, score_ixs=[0, 2], score_range=(-4., 4.), seq_start=0, seq_end=146, cse_end=76, top_n_isms=3, top_n_pwms=3, ts_diff_qtl=0.98, modisco_suffix='', save_figs=False, fig_name='apa_perturb_10') :
    
    fig_name += "_cell_type_2_ix_" + str(cell_type_2_ix)
    
    flat_mask_merged = np.max(np.concatenate([flat_masks[k][None, :] for k in range(len(flat_masks))], axis=0)[score_ixs, :], axis=0)
    
    #Distribution of model scores (split by distal/non-distal PASs)
    flat_ts_diff = (flat_ts_merged[:, cell_type_1_ix] - flat_ts_merged[:, cell_type_2_ix])[flat_mask_merged == 1.]
    
    if score_range is None :
        score_range = (np.min(flat_ts_diff), np.max(flat_ts_diff))
    
    f = plt.figure(figsize=(6, 4))
    
    plt.hist(flat_ts_diff, color='brown', alpha=0.5, bins=50, range=score_range, density=True)
    
    plt.xlabel("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "]", fontsize=12)
    plt.ylabel("Density of PASs", fontsize=12)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlim(score_range[0], score_range[1])
    
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + "_scores.png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_scores.eps")

    plt.show()
    
    #Plot top N most extremely perturbed PASs (model scores, measurements and ISMs)
    flat_ts_diff = flat_ts_merged[:, cell_type_2_ix] - flat_ts_merged[:, cell_type_1_ix]
    flat_y_diff = flat_y[:, cell_type_2_ix] - flat_y[:, cell_type_1_ix]

    #Sorted in ascending order of model scores (negative)
    min_ts_diff = np.quantile(flat_ts_diff[flat_mask_merged == 1.], q=1.-ts_diff_qtl)
    #print("min_ts_diff = " + str(round(min_ts_diff, 4)))

    outlier_index = np.nonzero((flat_ts_diff < min_ts_diff) & (flat_mask_merged == 1.))[0]

    outlier_ts_diff = flat_ts_diff[outlier_index]
    outlier_y_diff = flat_y_diff[outlier_index]

    outlier_index = outlier_index[np.argsort(outlier_y_diff)][:top_n_isms].tolist()

    print("- Top " + str(top_n_isms) + " ISMs (positive) - ")

    for i in outlier_index :

        print("Interpreting pattern " + str(i) + "...")
        print("Gene = " + str(flat_gene_names[i]))
        print("ID = " + str(flat_ids[i]))
        print("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_ts_diff[i], 4)))
        print("Y[" + subset_cell_types[cell_type_1_ix] + "] - Y[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_y_diff[i], 4)))

        score_str = "Prox" if flat_prox_mask[i] == 1. else ("Middle" if flat_middle_mask[i] == 1. else "Dist")
        print("[" + score_str + " site]")

        diff_scores = (flat_scores_merged[cell_type_1_ix, i, 0, seq_start:seq_end, :] - flat_scores_merged[cell_type_2_ix, i, 0, seq_start:seq_end, :]) * flat_x[i, 0, seq_start:seq_end, :]

        y_min = np.min(diff_scores) - 0.15 * np.max(np.abs(diff_scores))
        y_max = np.max(diff_scores) + 0.15 * np.max(np.abs(diff_scores))

        plot_seq_scores(
            diff_scores, y_min=y_min, y_max=y_max,
            figsize=(12, 1),
            plot_y_ticks=True,
            save_figs=save_figs,
            fig_name=fig_name + "_ism_pos" + "_example_ix_" + str(i)
        )

    #Plot top K modisco motifs
    modisco_f = h5py.File('polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_seq_start_0_seq_end_205_tfmodisco_out_cell_type_ix_1_' + str(cell_type_1_ix) + '_cell_type_ix_2_' + str(cell_type_2_ix) + '_score_ix_' + str(-1) + "" + modisco_suffix + '_no_rc.h5', 'r')

    fwd_pwms = []
    fwd_contribs = []
    fwd_hypo_contribs = []

    patterns = modisco_f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']

    for i in range(len(patterns) - 1) :
        fwd_pwms.append(patterns["pattern_" + str(i)]["sequence"]["fwd"][()])
        fwd_contribs.append(patterns["pattern_" + str(i)]["task_contrib_scores"]["fwd"][()])
        fwd_hypo_contribs.append(patterns["pattern_" + str(i)]["task_hypothetical_contribs"]["fwd"][()])

    all_pwms = []
    all_contribs = []
    all_hypo_contribs = []

    pwm_names = []

    for i in range(len(patterns) - 1) :
        all_pwms.append(fwd_pwms[i])
        all_contribs.append(fwd_contribs[i])
        all_hypo_contribs.append(fwd_hypo_contribs[i])

        pwm_names.append("pwm_" + str(i) + "_fwd")

    for i in range(min(top_n_pwms, len(all_pwms))) :
        print(pwm_names[i])
        plot_pwm_2(all_contribs[i], figsize=(8, 2), plot_y_ticks=False, save_figs=save_figs, fig_name=fig_name + "_pwm_pos" + "_motif_ix_" + str(i))

    print("")

    #Sorted in descending order of model scores (positive)
    min_ts_diff = np.quantile(flat_ts_diff[flat_mask_merged == 1.], q=ts_diff_qtl)
    #print("min_ts_diff = " + str(round(min_ts_diff, 4)))

    outlier_index = np.nonzero((flat_ts_diff > min_ts_diff) & (flat_mask_merged == 1.))[0]

    outlier_ts_diff = flat_ts_diff[outlier_index]
    outlier_y_diff = flat_y_diff[outlier_index]

    outlier_index = outlier_index[np.argsort(outlier_y_diff)[::-1]][:top_n_isms].tolist()

    print("- Top " + str(top_n_isms) + " ISMs (negative) - ")

    for i in outlier_index :

        print("Interpreting pattern " + str(i) + "...")
        print("Gene = " + str(flat_gene_names[i]))
        print("ID = " + str(flat_ids[i]))
        print("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_ts_diff[i], 4)))
        print("Y[" + subset_cell_types[cell_type_1_ix] + "] - Y[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_y_diff[i], 4)))

        score_str = "Prox" if flat_prox_mask[i] == 1. else ("Middle" if flat_middle_mask[i] == 1. else "Dist")
        print("[" + score_str + " site]")

        diff_scores = (flat_scores_merged[cell_type_1_ix, i, 0, seq_start:seq_end, :] - flat_scores_merged[cell_type_2_ix, i, 0, seq_start:seq_end, :]) * flat_x[i, 0, seq_start:seq_end, :]

        y_min = np.min(diff_scores) - 0.15 * np.max(np.abs(diff_scores))
        y_max = np.max(diff_scores) + 0.15 * np.max(np.abs(diff_scores))

        plot_seq_scores(
            diff_scores, y_min=y_min, y_max=y_max,
            figsize=(12, 1),
            plot_y_ticks=True,
            save_figs=save_figs,
            fig_name=fig_name + "_ism_neg" + "_example_ix_" + str(i)
        )

    #Plot top K modisco motifs
    modisco_f = h5py.File('polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_seq_start_0_seq_end_205_tfmodisco_out_cell_type_ix_1_' + str(cell_type_1_ix) + '_cell_type_ix_2_' + str(cell_type_2_ix) + '_score_ix_' + str(-1) + "_neg" + modisco_suffix + '_no_rc.h5', 'r')

    fwd_pwms = []
    fwd_contribs = []
    fwd_hypo_contribs = []

    patterns = modisco_f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']

    for i in range(len(patterns) - 1) :
        fwd_pwms.append(patterns["pattern_" + str(i)]["sequence"]["fwd"][()])
        fwd_contribs.append(patterns["pattern_" + str(i)]["task_contrib_scores"]["fwd"][()])
        fwd_hypo_contribs.append(patterns["pattern_" + str(i)]["task_hypothetical_contribs"]["fwd"][()])

    all_pwms = []
    all_contribs = []
    all_hypo_contribs = []

    pwm_names = []

    for i in range(len(patterns) - 1) :
        all_pwms.append(fwd_pwms[i])
        all_contribs.append(fwd_contribs[i])
        all_hypo_contribs.append(fwd_hypo_contribs[i])

        pwm_names.append("pwm_" + str(i) + "_fwd")

    for i in range(min(top_n_pwms, len(all_pwms))) :
        print(pwm_names[i])
        plot_pwm_2(all_contribs[i], figsize=(8, 2), plot_y_ticks=False, save_figs=save_figs, fig_name=fig_name + "_pwm_neg" + "_motif_ix_" + str(i))

    #Plot mean importance score across position
    mean_scores_curr = np.mean(flat_scores_merged[cell_type_1_ix, flat_mask_merged == 1., 0, :146, :] - flat_scores_merged[cell_type_2_ix, flat_mask_merged == 1., 0, :146, :], axis=(0, 2))
    
    f = plt.figure(figsize=(6, 4))
    
    plt.plot(np.arange(mean_scores_curr.shape[0])[:70], mean_scores_curr[:70], color='burlywood', linewidth=2)
    plt.plot(np.arange(mean_scores_curr.shape[0])[cse_end:146], mean_scores_curr[cse_end:146], color='burlywood', linewidth=2)

    plt.scatter(np.arange(mean_scores_curr.shape[0])[:70], mean_scores_curr[:70], color='brown', s=25)
    plt.scatter(np.arange(mean_scores_curr.shape[0])[cse_end:146], mean_scores_curr[cse_end:146], color='brown', s=25)

    plt.axvline(x=70, color='black', linewidth=3, linestyle='--')
    plt.axvline(x=76, color='black', linewidth=3, linestyle='--')

    plt.xlim(0, 145)
    #plt.ylim(0.)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.xticks([0, 70, 76, 146], ["-70bp", "CSE", "CSE+6bp", "+70bp"], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Mean ISM Score", fontsize=12)

    plt.title(str(subset_cell_types[cell_type_2_ix]), fontsize=12)
    
    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + "_ism_scores.png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_ism_scores.eps")

    plt.show()

def _print_interpretation_by_score(cell_type_1_ix=0, cell_type_2_ix=1, score_ixs=[0, 2], score_strs=['Prox', 'Dist'], score_colors=['green', 'red'], score_light_colors=['lightgreen', 'lightcoral'], score_range=(-4., 4.), seq_start=0, seq_end=146, cse_end=76, top_n_isms=3, top_n_pwms=3, ts_diff_qtl=0.98, modisco_suffix='', save_figs=False, fig_name='apa_perturb_10_by_score') :
    
    fig_name += "_cell_type_2_ix_" + str(cell_type_2_ix)
    
    #Distribution of model scores (split by distal/non-distal PASs)
    flat_ts_diffs = [
        (flat_ts[:, cell_type_1_ix, score_ix] - flat_ts[:, cell_type_2_ix, score_ix])[flat_masks[score_ix] == 1.]
        for score_ix in score_ixs
    ]
    
    if score_range is None :
        score_range = (np.min(flat_ts_diffs), np.max(flat_ts_diffs))
    
    f = plt.figure(figsize=(6, 4))
    
    for score_i, score_ix in enumerate(score_ixs) :
        plt.hist(flat_ts_diffs[score_i], color=score_colors[score_i], alpha=0.5, bins=50, range=score_range, density=True, label=score_strs[score_i])
    
    plt.xlabel("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "]", fontsize=12)
    plt.ylabel("Density of PASs", fontsize=12)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlim(score_range[0], score_range[1])
    
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + "_scores.png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_scores.eps")

    plt.show()
    
    #Plot example PAS ISM maps and scores
    for score_i, score_ix in enumerate(score_ixs) :
        
        print("[score_ix = " + str(score_ix) + "]")
    
        #Plot top N most extremely perturbed PASs (model scores, measurements and ISMs)
        flat_ts_diff = flat_ts[:, cell_type_2_ix, score_ix] - flat_ts[:, cell_type_1_ix, score_ix]
        flat_y_diff = flat_y[:, cell_type_2_ix] - flat_y[:, cell_type_1_ix]

        #Sorted in ascending order of model scores (negative)
        min_ts_diff = np.quantile(flat_ts_diff[flat_masks[score_ix] == 1.], q=1.-ts_diff_qtl)
        #print("min_ts_diff = " + str(round(min_ts_diff, 4)))

        outlier_index = np.nonzero((flat_ts_diff < min_ts_diff) & (flat_masks[score_ix] == 1.))[0]

        outlier_ts_diff = flat_ts_diff[outlier_index]
        outlier_y_diff = flat_y_diff[outlier_index]

        outlier_index = outlier_index[np.argsort(outlier_y_diff)][:top_n_isms].tolist()

        print("- Top " + str(top_n_isms) + " ISMs (positive) - ")

        for i in outlier_index :

            print("Interpreting pattern " + str(i) + "...")
            print("Gene = " + str(flat_gene_names[i]))
            print("ID = " + str(flat_ids[i]))
            print("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_ts_diff[i], 4)))
            print("Y[" + subset_cell_types[cell_type_1_ix] + "] - Y[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_y_diff[i], 4)))
            print("[" + score_strs[score_i] + " site]")
            
            diff_scores = (flat_scores[cell_type_1_ix, :, i, 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, :, i, 0, seq_start:seq_end, :]) * flat_x[i, 0, seq_start:seq_end, :]
            
            y_min = np.min(diff_scores[score_ix]) - 0.15 * np.max(np.abs(diff_scores[score_ix]))
            y_max = np.max(diff_scores[score_ix]) + 0.15 * np.max(np.abs(diff_scores[score_ix]))

            plot_seq_scores(
                diff_scores[score_ix], y_min=y_min, y_max=y_max,
                figsize=(12, 1),
                plot_y_ticks=True,
                save_figs=save_figs,
                fig_name=fig_name + "_ism_score_ix_" + str(score_ix) + "_pos" + "_example_ix_" + str(i)
            )
        
        #Plot top K modisco motifs
        modisco_f = h5py.File('polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_seq_start_0_seq_end_205_tfmodisco_out_cell_type_ix_1_' + str(cell_type_1_ix) + '_cell_type_ix_2_' + str(cell_type_2_ix) + '_score_ix_' + str(score_ix) + "" + modisco_suffix + '_no_rc.h5', 'r')

        fwd_pwms = []
        fwd_contribs = []
        fwd_hypo_contribs = []

        patterns = modisco_f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']

        for i in range(len(patterns) - 1) :
            fwd_pwms.append(patterns["pattern_" + str(i)]["sequence"]["fwd"][()])
            fwd_contribs.append(patterns["pattern_" + str(i)]["task_contrib_scores"]["fwd"][()])
            fwd_hypo_contribs.append(patterns["pattern_" + str(i)]["task_hypothetical_contribs"]["fwd"][()])

        all_pwms = []
        all_contribs = []
        all_hypo_contribs = []

        pwm_names = []

        for i in range(len(patterns) - 1) :
            all_pwms.append(fwd_pwms[i])
            all_contribs.append(fwd_contribs[i])
            all_hypo_contribs.append(fwd_hypo_contribs[i])

            pwm_names.append("pwm_" + str(i) + "_fwd")

        for i in range(min(top_n_pwms, len(all_pwms))) :
            print(pwm_names[i])
            plot_pwm_2(all_contribs[i], figsize=(8, 2), plot_y_ticks=False, save_figs=save_figs, fig_name=fig_name + "_pwm_score_ix_" + str(score_ix) + "_pos" + "_motif_ix_" + str(i))

        print("")

        #Sorted in descending order of model scores (positive)
        min_ts_diff = np.quantile(flat_ts_diff[flat_masks[score_ix] == 1.], q=ts_diff_qtl)
        #print("min_ts_diff = " + str(round(min_ts_diff, 4)))

        outlier_index = np.nonzero((flat_ts_diff > min_ts_diff) & (flat_masks[score_ix] == 1.))[0]

        outlier_ts_diff = flat_ts_diff[outlier_index]
        outlier_y_diff = flat_y_diff[outlier_index]

        outlier_index = outlier_index[np.argsort(outlier_y_diff)[::-1]][:top_n_isms].tolist()

        print("- Top " + str(top_n_isms) + " ISMs (negative) - ")

        for i in outlier_index :

            print("Interpreting pattern " + str(i) + "...")
            print("Gene = " + str(flat_gene_names[i]))
            print("ID = " + str(flat_ids[i]))
            print("Score[" + subset_cell_types[cell_type_1_ix] + "] - Score[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_ts_diff[i], 4)))
            print("Y[" + subset_cell_types[cell_type_1_ix] + "] - Y[" + subset_cell_types[cell_type_2_ix] + "] = " + str(round(-flat_y_diff[i], 4)))
            print("[" + score_strs[score_i] + " site]")
            
            diff_scores = (flat_scores[cell_type_1_ix, :, i, 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, :, i, 0, seq_start:seq_end, :]) * flat_x[i, 0, seq_start:seq_end, :]
            
            y_min = np.min(diff_scores[score_ix]) - 0.15 * np.max(np.abs(diff_scores[score_ix]))
            y_max = np.max(diff_scores[score_ix]) + 0.15 * np.max(np.abs(diff_scores[score_ix]))

            plot_seq_scores(
                diff_scores[score_ix], y_min=y_min, y_max=y_max,
                figsize=(12, 1),
                plot_y_ticks=True,
                save_figs=save_figs,
                fig_name=fig_name + "_ism_score_ix_" + str(score_ix) + "_neg" + "_example_ix_" + str(i)
            )
            
        #Plot top K modisco motifs
        modisco_f = h5py.File('polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_seq_start_0_seq_end_205_tfmodisco_out_cell_type_ix_1_' + str(cell_type_1_ix) + '_cell_type_ix_2_' + str(cell_type_2_ix) + '_score_ix_' + str(score_ix) + "_neg" + modisco_suffix + '_no_rc.h5', 'r')

        fwd_pwms = []
        fwd_contribs = []
        fwd_hypo_contribs = []

        patterns = modisco_f['metacluster_idx_to_submetacluster_results']['metacluster_0']['seqlets_to_patterns_result']['patterns']

        for i in range(len(patterns) - 1) :
            fwd_pwms.append(patterns["pattern_" + str(i)]["sequence"]["fwd"][()])
            fwd_contribs.append(patterns["pattern_" + str(i)]["task_contrib_scores"]["fwd"][()])
            fwd_hypo_contribs.append(patterns["pattern_" + str(i)]["task_hypothetical_contribs"]["fwd"][()])

        all_pwms = []
        all_contribs = []
        all_hypo_contribs = []

        pwm_names = []

        for i in range(len(patterns) - 1) :
            all_pwms.append(fwd_pwms[i])
            all_contribs.append(fwd_contribs[i])
            all_hypo_contribs.append(fwd_hypo_contribs[i])

            pwm_names.append("pwm_" + str(i) + "_fwd")

        for i in range(min(top_n_pwms, len(all_pwms))) :
            print(pwm_names[i])
            plot_pwm_2(all_contribs[i], figsize=(8, 2), plot_y_ticks=False, save_figs=save_figs, fig_name=fig_name + "_pwm_score_ix_" + str(score_ix) + "_neg" + "_motif_ix_" + str(i))

    #Plot mean importance score across position (distal/non-distal PASs)
    mean_scores = [
        np.mean(flat_scores[cell_type_1_ix, score_ix, flat_masks[score_ix] == 1., 0, :146, :] - flat_scores[cell_type_2_ix, score_ix, flat_masks[score_ix] == 1., 0, :146, :], axis=(0, 2))
        for score_ix in score_ixs
    ]
    
    f = plt.figure(figsize=(6, 4))
    
    for score_i, score_ix in enumerate(mean_scores) :
        mean_scores_curr = mean_scores[score_i]
        
        plt.plot(np.arange(mean_scores_curr.shape[0])[:70], mean_scores_curr[:70], color=score_light_colors[score_i], linewidth=2)
        plt.plot(np.arange(mean_scores_curr.shape[0])[cse_end:146], mean_scores_curr[cse_end:146], color=score_light_colors[score_i], linewidth=2)

        plt.scatter(np.arange(mean_scores_curr.shape[0])[:70], mean_scores_curr[:70], color=score_colors[score_i], s=25, label="(" + score_strs[score_i] + ")")
        plt.scatter(np.arange(mean_scores_curr.shape[0])[cse_end:146], mean_scores_curr[cse_end:146], color=score_colors[score_i], s=25)

    plt.axvline(x=70, color='black', linewidth=3, linestyle='--')
    plt.axvline(x=76, color='black', linewidth=3, linestyle='--')

    plt.xlim(0, 145)
    #plt.ylim(0.)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.xticks([0, 70, 76, 146], ["-70bp", "CSE", "CSE+6bp", "+70bp"], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Mean ISM Score", fontsize=12)

    plt.title(str(subset_cell_types[cell_type_2_ix]), fontsize=12)
    
    plt.legend(fontsize=12)

    plt.tight_layout()
    
    if save_figs :
        plt.savefig(fig_name + "_ism_scores.png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_ism_scores.eps")

    plt.show()

#Code to perform and visualize variant interpretations

def _predict(tissue_models, x) :
    
    y_hats = []
    for bootstrap_ix in range(len(tissue_models)) :
        y_hats.append(tissue_models[bootstrap_ix].predict(x=[x], batch_size=32)[..., None])
    
    return np.mean(np.concatenate(y_hats, axis=-1), axis=-1)

def _interpret(tissue_models, x) :
    
    #Perform full window-shuffled ISM on PAS sequences

    mean_g_importance_scores = np.zeros((n_cell_types, 3, x.shape[0], 1, 205, 4))

    for bootstrap_ix in range(len(tissue_models)) :

        importance_scores, _ = _ism_shuffle(
            tissue_models[bootstrap_ix],
            x,
            n_out=n_cell_types,
            n_samples=32,
            window_size=5
        )

        importance_scores = -importance_scores
        g_importance_scores = importance_scores * x[None, None, ...]

        mean_g_importance_scores += g_importance_scores

        importance_scores = None
        g_importance_scores = None

    mean_g_importance_scores /= float(n_bootstraps)

    return mean_g_importance_scores


def _scramble_sequence(seq, start, end) :
    
    seq_index = np.arange(len(seq), dtype=np.int32)
    
    np.random.shuffle(seq_index[start:end])
    
    new_seq = ""
    for j in range(len(seq)) :
        new_seq += seq[seq_index[j]]
    
    return new_seq

def _onehot_encode(seq_ref) :
    
    flat_x_ref = np.zeros((1, 1, 205, 4))
    for j in range(len(seq_ref)) :
        if seq_ref[j] == 'A' :
            flat_x_ref[0, 0, j, 0] = 1.
        elif seq_ref[j] == 'C' :
            flat_x_ref[0, 0, j, 1] = 1.
        elif seq_ref[j] == 'G' :
            flat_x_ref[0, 0, j, 2] = 1.
        elif seq_ref[j] == 'T' :
            flat_x_ref[0, 0, j, 3] = 1.
    
    return flat_x_ref

def _get_null_onehots(seq_ref, seq_var, find_neutral=False, cell_type_1_ix=0, cell_type_2_ix=4, score_ix=2, rand_pos=False, n=1000, scramble_start=76, scramble_end=126) :
    
    #Get flat representation of sequences as dataframe

    if n is None :
        n = flat_x.shape[0]
    
    seqs = flat_x[:, 0, ...]

    if find_neutral :
        
        cost = -flat_ts[:, cell_type_1_ix, score_ix] + flat_ts[:, cell_type_2_ix, score_ix]
        sort_index = np.argsort(cost)
        
        seqs = np.copy(seqs[sort_index])
    
    seqs_str = []
    for i in range(seqs.shape[0]) :
        seq = ""
        for j in range(seqs.shape[1]) :
            if seqs[i, j, 0] == 1. :
                seq += "A"
            elif seqs[i, j, 1] == 1. :
                seq += "C"
            elif seqs[i, j, 2] == 1. :
                seq += "G"
            elif seqs[i, j, 3] == 1. :
                seq += "T"

        seqs_str.append(seq)

    seqs_str = np.array(seqs_str, dtype=np.object)

    df = pd.DataFrame({
        "seq_ref" : seqs_str.tolist(),
    }).iloc[:n].copy().reset_index(drop=True)
    
    df['i'] = np.arange(n, dtype=np.int32)
    
    mut_pos = -1
    mut_ref = 'X'
    mut_var = 'X'
    
    for j in range(len(seq_ref)) :
        if seq_ref[j] != seq_var[j] :
            mut_pos = j
            mut_ref = seq_ref[j]
            mut_var = seq_var[j]
            break

    mut_poses = np.zeros(n, dtype=np.int32)
    if not rand_pos :
        mut_poses[:] = mut_pos
    else :
        mut_poses[:] = np.random.choice((np.arange(scramble_end-scramble_start) + scramble_start).tolist(), size=(n,)).tolist()
    
    df['seq_var'] = df.apply(lambda row: row['seq_ref'][:mut_poses[row['i']]] + mut_var + row['seq_ref'][mut_poses[row['i']]+1:], axis=1)
    df['seq_ref'] = df.apply(lambda row: row['seq_ref'][:mut_poses[row['i']]] + mut_ref + row['seq_ref'][mut_poses[row['i']]+1:], axis=1)
    
    seq_refs = df['seq_ref'].values.tolist()
    seq_vars = df['seq_var'].values.tolist()
    
    seq_refs_scrambled = []
    seq_vars_scrambled = []
    for i, seq_ref in enumerate(seq_refs) :
        seq_ref_scrambled = _scramble_sequence(seq_ref, scramble_start, scramble_end)
        
        #Make sure the CSE remains unscrambled
        seq_ref_scrambled = seq_ref_scrambled[:70] + seq_ref[70:76] + seq_ref_scrambled[76:]
        
        seq_refs_scrambled.append(seq_ref_scrambled[:mut_poses[i]] + mut_ref + seq_ref_scrambled[mut_poses[i]+1:])
        seq_vars_scrambled.append(seq_ref_scrambled[:mut_poses[i]] + mut_var + seq_ref_scrambled[mut_poses[i]+1:])
    
    #One-hot-encode
    seq_refs = np.concatenate([_onehot_encode(seq_ref) for seq_ref in seq_refs], axis=0)
    seq_vars = np.concatenate([_onehot_encode(seq_var) for seq_var in seq_vars], axis=0)
    
    seq_refs_scrambled = np.concatenate([_onehot_encode(seq_ref) for seq_ref in seq_refs_scrambled], axis=0)
    seq_vars_scrambled = np.concatenate([_onehot_encode(seq_var) for seq_var in seq_vars_scrambled], axis=0)
    
    return seq_refs, seq_vars, seq_refs_scrambled, seq_vars_scrambled

def _interpret_mutant(seq_ref, seq_var, cell_type_1_ix=0, cell_type_2_ix=4, score_ix=2, find_neutral=False, rand_pos=True, n=1000, scramble_start=76, scramble_end=146, seq_start=0, seq_end=146, lof_or_gof='gof', fixed_y_min=-0.1, save_figs=False, fig_name='default') :

    #One-hot-encode ref sequence
    flat_x_ref = _onehot_encode(seq_ref)

    #One-hot-encode var sequence
    flat_x_var = _onehot_encode(seq_var)
    
    #Get one-hot-coded null pairs
    null_x_ref, null_x_var, null_x_ref_s, null_x_var_s = _get_null_onehots(seq_ref, seq_var, find_neutral=find_neutral, cell_type_1_ix=cell_type_1_ix, cell_type_2_ix=cell_type_2_ix, score_ix=score_ix, rand_pos=rand_pos, n=n, scramble_start=scramble_start, scramble_end=scramble_end)

    #Get predictions
    s_ref = _predict(tissue_models, flat_x_ref)
    s_var = _predict(tissue_models, flat_x_var)
    
    s_ref_null = _predict(tissue_models, null_x_ref)
    s_var_null = _predict(tissue_models, null_x_var)
    s_ref_null_s = _predict(tissue_models, null_x_ref_s)
    s_var_null_s = _predict(tissue_models, null_x_var_s)
    
    #Get ISM maps
    ism_ref = _interpret(tissue_models, flat_x_ref)
    ism_var = _interpret(tissue_models, flat_x_var)
    
    #Compute variant effects
    s_ref_diff = s_ref[0, cell_type_1_ix, score_ix] - s_ref[0, cell_type_2_ix, score_ix]
    s_var_diff = s_var[0, cell_type_1_ix, score_ix] - s_var[0, cell_type_2_ix, score_ix]
    
    s_ref_diff_null = s_ref_null[:, cell_type_1_ix, score_ix] - s_ref_null[:, cell_type_2_ix, score_ix]
    s_var_diff_null = s_var_null[:, cell_type_1_ix, score_ix] - s_var_null[:, cell_type_2_ix, score_ix]
    s_ref_diff_null_s = s_ref_null_s[:, cell_type_1_ix, score_ix] - s_ref_null_s[:, cell_type_2_ix, score_ix]
    s_var_diff_null_s = s_var_null_s[:, cell_type_1_ix, score_ix] - s_var_null_s[:, cell_type_2_ix, score_ix]

    s_delta = s_var_diff - s_ref_diff
    
    s_delta_null = s_var_diff_null - s_ref_diff_null
    s_delta_null_s = s_var_diff_null_s - s_ref_diff_null_s
    
    #Compute variant effects (all perturbations)
    s_ref_diff_all = s_ref[0, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_ref[0, cell_type_1_ix+1:, score_ix]
    s_var_diff_all = s_var[0, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_var[0, cell_type_1_ix+1:, score_ix]
    
    s_ref_diff_null_all = s_ref_null[:, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_ref_null[:, cell_type_1_ix+1:, score_ix]
    s_var_diff_null_all = s_var_null[:, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_var_null[:, cell_type_1_ix+1:, score_ix]
    s_ref_diff_null_s_all = s_ref_null_s[:, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_ref_null_s[:, cell_type_1_ix+1:, score_ix]
    s_var_diff_null_s_all = s_var_null_s[:, cell_type_1_ix:cell_type_1_ix+1, score_ix] - s_var_null_s[:, cell_type_1_ix+1:, score_ix]

    s_delta_all = s_var_diff_all - s_ref_diff_all
    
    s_delta_null_all = s_var_diff_null_all - s_ref_diff_null_all
    s_delta_null_s_all = s_var_diff_null_s_all - s_ref_diff_null_s_all
    
    s_delta_null_all = [s_delta_null_all[:, cell_type_ix] for cell_type_ix in range(s_delta_null_all.shape[1])]
    s_delta_null_s_all = [s_delta_null_s_all[:, cell_type_ix] for cell_type_ix in range(s_delta_null_s_all.shape[1])]
    
    #Visualize variant effect interpretation
    ism_ref_diff = (ism_ref[cell_type_1_ix, score_ix, 0, 0, seq_start:seq_end, :] - ism_ref[cell_type_2_ix, score_ix, 0, 0, seq_start:seq_end, :]) * flat_x_ref[0, 0, seq_start:seq_end, :]
    ism_var_diff = (ism_var[cell_type_1_ix, score_ix, 0, 0, seq_start:seq_end, :] - ism_var[cell_type_2_ix, score_ix, 0, 0, seq_start:seq_end, :]) * flat_x_var[0, 0, seq_start:seq_end, :]

    y_min = min(
        np.min(ism_ref_diff) - 0.1 * np.max(np.abs(ism_ref_diff)),
        np.min(ism_var_diff) - 0.1 * np.max(np.abs(ism_var_diff))
    )
    y_max = max(
        np.max(ism_ref_diff) + 0.1 * np.max(np.abs(ism_ref_diff)),
        np.max(ism_var_diff) + 0.1 * np.max(np.abs(ism_var_diff))
    )
    print("ref score = " + str(round(s_ref_diff, 4)))
    plot_seq_scores(
        ism_ref_diff,
        figsize=(12, 1),
        y_min=y_min if fixed_y_min is None else fixed_y_min, y_max=y_max,
        plot_y_ticks=True,
        save_figs=save_figs,
        fig_name=fig_name + "_ism_cell_type_2_ix_" + str(cell_type_2_ix) + "_ref"
    )

    print("var score = " + str(round(s_var_diff, 4)))
    plot_seq_scores(
        ism_var_diff,
        figsize=(12, 1),
        y_min=y_min if fixed_y_min is None else fixed_y_min, y_max=y_max,
        plot_y_ticks=True,
        save_figs=save_figs,
        fig_name=fig_name + "_ism_cell_type_2_ix_" + str(cell_type_2_ix) + "_var"
    )
    
    print("delta = " + str(round(s_delta, 4)))
    
    if lof_or_gof == 'gof' :
        ism_delta = np.sum(ism_var_diff - ism_ref_diff, axis=-1, keepdims=True) * flat_x_var[0, 0, seq_start:seq_end, :]

        y_min = np.min(ism_delta) - 0.1 * np.max(np.abs(ism_delta))
        y_max = np.max(ism_delta) + 0.1 * np.max(np.abs(ism_delta))

        plot_seq_scores(
            ism_delta,
            figsize=(12, 1),
            y_min=y_min if fixed_y_min is None else fixed_y_min, y_max=y_max,
            plot_y_ticks=True,
            save_figs=save_figs,
            fig_name=fig_name + "_ism_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta"
        )
    else :
        ism_delta = np.sum(ism_var_diff - ism_ref_diff, axis=-1, keepdims=True) * flat_x_ref[0, 0, seq_start:seq_end, :]

        y_min = np.min(ism_delta) - 0.1 * np.max(np.abs(ism_delta))
        y_max = np.max(ism_delta) + 0.1 * np.max(np.abs(ism_delta))

        plot_seq_scores(
            ism_delta,
            figsize=(12, 1),
            y_min=y_min if fixed_y_min is not None else None, y_max=-fixed_y_min if fixed_y_min is not None else None,
            plot_y_ticks=True,
            save_figs=save_figs,
            fig_name=fig_name + "_ism_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta"
        )
    
    print("Null distributions:")
    
    import seaborn as sns
    
    f = plt.figure(figsize=(3, 4))
    
    p_val = 1. - np.sum((s_delta > s_delta_null) if lof_or_gof == 'gof' else (s_delta < s_delta_null)) / s_delta_null.shape[0]
    
    print("Empirical p = " + str(round(p_val, 3)))

    sns.stripplot(data=[s_delta_null], alpha=0.5, jitter=0.25, palette=['black'])
    sns.boxplot(data=[s_delta_null], linewidth=2, fliersize=0., palette=['lightgray'])

    l1 = plt.axhline(y=s_delta, color='red', linewidth=3, label='Mutation')
    
    plt.xticks([0], ["Null Mut (WT)"], fontsize=12)

    plt.yticks(fontsize=12)
    plt.ylabel("LOR[Var, Ref] (NT - Perturb)", fontsize=12)
    
    plt.legend(handles=[l1], fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_score_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta_null" + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_score_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta_null" + ".eps")
    
    plt.show()
    
    f = plt.figure(figsize=(3, 4))

    p_val_s = 1. - np.sum((s_delta > s_delta_null_s) if lof_or_gof == 'gof' else (s_delta < s_delta_null_s)) / s_delta_null_s.shape[0]
    
    print("Empirical p = " + str(round(p_val_s, 3)) + " (scrambled)")
    
    sns.stripplot(data=[s_delta_null_s], alpha=0.5, jitter=0.25, palette=['black'])
    sns.boxplot(data=[s_delta_null_s], linewidth=2, fliersize=0., palette=['lightgray'])

    l1 = plt.axhline(y=s_delta, color='red', linewidth=3, label='Mutation')
    
    plt.xticks([0], ["Null Mut (Scrambled)"], fontsize=12)

    plt.yticks(fontsize=12)
    plt.ylabel("LOR[Var, Ref] (NT - Perturb)", fontsize=12)
    
    plt.legend(handles=[l1], fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_score_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta_null_s" + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_score_cell_type_2_ix_" + str(cell_type_2_ix) + "_delta_null_s" + ".eps")
    
    plt.show()

    print("---")
    
    print("Null distributions (all perturbations):")
    
    import seaborn as sns
    
    f = plt.figure(figsize=(9, 4))
    
    p_vals = []
    for cell_type_ix in range(len(s_delta_null_all)) :
        p_vals.append(round(1. - np.sum((s_delta_all[cell_type_ix] > s_delta_null_all[cell_type_ix]) if lof_or_gof == 'gof' else (s_delta_all[cell_type_ix] < s_delta_null_all[cell_type_ix])) / s_delta_null_all[cell_type_ix].shape[0], 3))
    
    print("cell_types = " + str(subset_cell_types[1:].tolist()))
    print("Empirical p = " + str(p_vals))

    sns.stripplot(data=s_delta_null_all, alpha=0.5, jitter=0.25, palette='magma')
    sns.boxplot(data=s_delta_null_all, linewidth=2, fliersize=0., palette='magma')

    for cell_type_ix in range(len(s_delta_null_all)) :
        plt.plot([cell_type_ix-0.5, cell_type_ix+0.5], [s_delta_all[cell_type_ix], s_delta_all[cell_type_ix]], color='red', linewidth=3)
    
    plt.xticks(np.arange(len(s_delta_null_all)), subset_cell_types[1:].tolist(), rotation=45, fontsize=12)

    plt.yticks(fontsize=12)
    plt.ylabel("LOR[Var, Ref] (NT - Perturb)", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_score_all_cell_types_delta_null" + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_score_all_cell_types_delta_null" + ".eps")
    
    plt.show()
    
    f = plt.figure(figsize=(9, 4))
    
    p_vals = []
    for cell_type_ix in range(len(s_delta_null_s_all)) :
        p_vals.append(round(1. - np.sum((s_delta_all[cell_type_ix] > s_delta_null_s_all[cell_type_ix]) if lof_or_gof == 'gof' else (s_delta_all[cell_type_ix] < s_delta_null_s_all[cell_type_ix])) / s_delta_null_s_all[cell_type_ix].shape[0], 3))
    
    print("cell_types = " + str(subset_cell_types[1:].tolist()))
    print("Empirical p = " + str(p_vals))

    sns.stripplot(data=s_delta_null_s_all, alpha=0.5, jitter=0.25, palette='magma')
    sns.boxplot(data=s_delta_null_s_all, linewidth=2, fliersize=0., palette='magma')

    for cell_type_ix in range(len(s_delta_null_s_all)) :
        plt.plot([cell_type_ix-0.5, cell_type_ix+0.5], [s_delta_all[cell_type_ix], s_delta_all[cell_type_ix]], color='red', linewidth=3)
    
    plt.xticks(np.arange(len(s_delta_null_s_all)), subset_cell_types[1:].tolist(), rotation=45, fontsize=12)

    plt.yticks(fontsize=12)
    plt.ylabel("LOR[Var, Ref] (NT - Perturb)", fontsize=12)

    plt.tight_layout()

    if save_figs :
        plt.savefig(fig_name + "_score_all_cell_types_delta_null_s" + ".png", transparent=True, dpi=600)
        plt.savefig(fig_name + "_score_all_cell_types_delta_null_s" + ".eps")
    
    plt.show()
