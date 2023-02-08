import os
import sys
import pandas as pd

import numpy as np
import scipy

def run_modisco(cell_type_2_ix, score_ix, sign_multiplier, filter_pases) :
    
    sign_suffix = ''
    if sign_multiplier == -1 :
        sign_suffix = '_neg'
    
    filter_suffix = ''
    if filter_pases == 1 :
        filter_suffix = '_filtered'
    
    #Load Data
    df = pd.read_csv('polyadb_features_pas_3_utr3_perturb.csv', sep='\t')

    save_dict = np.load("polyadb_features_pas_3_utr3_perturb.npz")
    x, m, l, c, y = save_dict['x'], save_dict['m'], save_dict['l'], save_dict['c'], save_dict['y']

    print("x.shape = " + str(x.shape))
    print("m.shape = " + str(m.shape))
    print("l.shape = " + str(l.shape))
    print("c.shape = " + str(c.shape))
    print("y.shape = " + str(y.shape))
    
    #Distal PAS indices and masks
    dist_index = np.array([np.nonzero(m[i, :])[0][-1] for i in range(m.shape[0])])

    dist_mask = np.zeros(m.shape)
    for i in range(m.shape[0]) :
        dist_mask[i, dist_index[i]] = 1.
    
    #Load tissue-specific PAS model and generate scores for select tissue types

    subset_cell_types = np.array([
        'NT',
        'CPSF4',
        'CPSF6',
        'CSTF1',
        'CSTF3',
        'FIP1L1',
        'NUDT21',
        'RBBP6',
        'SRSF3',
        'SYMPK',
        'THOC5'
    ], dtype=np.object)

    subset_cell_type_dict = {
        cell_type : cell_type_i for cell_type_i, cell_type in enumerate(subset_cell_types)
    }
    
    #Define tissue-/cell- types

    cell_types = np.array([
        'rpm',
        'NT',
        'CDC73',
        'CPSF1',
        'CPSF2',
        'CPSF3',
        'CPSF3L',
        'CPSF4',
        'CPSF6',
        'CSTF1',
        'CSTF3',
        'CTR9',
        'FIP1L1',
        'LEO1',
        'NUDT21',
        'PABPC1',
        'PABPN1',
        'PAF1',
        'PAPOLA',
        'PCF11',
        'RBBP6',
        'RPRD1A',
        'RPRD1B',
        'SCAF8',
        'SF3A1',
        'SRSF3',
        'SYMPK',
        'THOC5'
    ], dtype=np.object)

    cell_type_dict = {
        cell_type : cell_type_i for cell_type_i, cell_type in enumerate(cell_types)
    }
        
    flat_x = np.reshape(x, (x.shape[0] * x.shape[1], 1, 205, 4))
    flat_m = np.reshape(m, (x.shape[0] * x.shape[1],))
    flat_dist_mask = np.reshape(dist_mask, (x.shape[0] * x.shape[1],))
    flat_gene_ind = np.reshape(np.tile(np.arange(x.shape[0])[:, None], (1, x.shape[1])), (x.shape[0] * x.shape[1],))
    flat_pas_ind = np.reshape(np.tile(np.arange(x.shape[1])[None, :], (x.shape[0], 1)), (x.shape[0] * x.shape[1],))
    
    flat_keep_index = np.nonzero(flat_m >= 1)[0]

    flat_x = flat_x[flat_keep_index, ...]
    flat_m = flat_m[flat_keep_index, ...]
    flat_dist_mask = flat_dist_mask[flat_keep_index, ...]
    flat_gene_ind = flat_gene_ind[flat_keep_index, ...]
    flat_pas_ind = flat_pas_ind[flat_keep_index, ...]
    
    #Construct masks for proximal/middle/distal sites

    flat_prox_mask = np.array((flat_pas_ind == 0), dtype=np.float32)
    flat_middle_mask = 1. - flat_dist_mask - flat_prox_mask
    
    flat_masks = [
        flat_prox_mask,
        flat_middle_mask,
        flat_dist_mask,
    ]

    #Re-load gated importance scores

    flat_scores = np.load("polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_flat_g_scores.npy")
    flat_scores = np.tile(flat_scores, (1, 1, 1, 1, 1, 4)) * flat_x[None, None, ...]
    
    n_cell_types = subset_cell_types.shape[0]

    print("n_cell_types = " + str(n_cell_types))
    
    #Run modisco (for specified perturbation)

    import modisco
    import h5py

    seq_start = 0
    seq_end = 205

    cell_type_1_ix = 0

    print("cell_type_1 = '" + str(subset_cell_types[cell_type_1_ix]) + "'")
    print("cell_type_2 = '" + str(subset_cell_types[cell_type_2_ix]) + "'")
    
    print("score_ix = " + str(score_ix))
    print("sign_multiplier = " + str(sign_multiplier) + " ('" + sign_suffix + "')")
    print("filter_pases = " + str(filter_pases) + " ('" + filter_suffix + "')")

    seqs = flat_x[:, 0, seq_start:seq_end, :]
    scores = (flat_scores[cell_type_1_ix, score_ix, :, 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, score_ix, :, 0, seq_start:seq_end, :]) * flat_x[:, 0, seq_start:seq_end, :]
    if filter_pases == 1 and score_ix != -1 :
        seqs = flat_x[flat_masks[score_ix] == 1., 0, seq_start:seq_end, :]
        scores = (flat_scores[cell_type_1_ix, score_ix, flat_masks[score_ix] == 1., 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, score_ix, flat_masks[score_ix] == 1., 0, seq_start:seq_end, :]) * flat_x[flat_masks[score_ix] == 1., 0, seq_start:seq_end, :]
    elif score_ix == -1 :
        seqs = flat_x[:, 0, seq_start:seq_end, :]
        scores = np.zeros(scores.shape)
        
        scores[flat_masks[0] == 1., ...] = (flat_scores[cell_type_1_ix, 0, flat_masks[0] == 1., 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, 0, flat_masks[0] == 1., 0, seq_start:seq_end, :]) * flat_x[flat_masks[0] == 1., 0, seq_start:seq_end, :]
        scores[flat_masks[1] == 1., ...] = (flat_scores[cell_type_1_ix, 1, flat_masks[1] == 1., 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, 1, flat_masks[1] == 1., 0, seq_start:seq_end, :]) * flat_x[flat_masks[1] == 1., 0, seq_start:seq_end, :]
        scores[flat_masks[2] == 1., ...] = (flat_scores[cell_type_1_ix, 2, flat_masks[2] == 1., 0, seq_start:seq_end, :] - flat_scores[cell_type_2_ix, 2, flat_masks[2] == 1., 0, seq_start:seq_end, :]) * flat_x[flat_masks[2] == 1., 0, seq_start:seq_end, :]
    
    print("seqs.shape = " + str(seqs.shape))
    print("scores.shape = " + str(scores.shape))
    
    ##########################
    # run tfmodisco workflow #
    ##########################
    tfm_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                      sliding_window_size=6,
                      flank_size=2,
                      target_seqlet_fdr=0.1,
                      max_seqlets_per_metacluster=40000, # don't put constrain on this
                      separate_pos_neg_thresholds=True,
                      seqlets_to_patterns_factory=
                          modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                              n_cores=16, # use 16 cores
                              trim_to_window_size=8,
                              initial_flank_to_add=1,
                              kmer_len=5, num_gaps=2,
                              num_mismatches=1,
                              final_min_cluster_size=20
                          )
                      )(
                    task_names=["task"], #list(target_importance.keys()),
                    contrib_scores={"task": sign_multiplier * seqs * scores}, #target_importance,
                    hypothetical_contribs={"task": sign_multiplier * scores}, #target_hypothetical,
                    revcomp=False,
                    one_hot=seqs)

    h5_out = h5py.File('polyadb_features_pas_3_utr3_perturb_resnet_covar_drop_seq_start_' + str(seq_start) + '_seq_end_' + str(seq_end) + '_tfmodisco_out_cell_type_ix_1_' + str(cell_type_1_ix) + '_cell_type_ix_2_' + str(cell_type_2_ix) + '_score_ix_' + str(score_ix) + sign_suffix + filter_suffix + '_no_rc.h5', 'w')
    tfm_results.save_hdf5(h5_out)
    h5_out.close()
    
    return

if __name__ == "__main__" :
    
    cell_type_2_ix = int(sys.argv[1])
    score_ix = int(sys.argv[2])
    sign_multiplier = int(sys.argv[3])
    filter_pases = int(sys.argv[4])
    
    run_modisco(cell_type_2_ix, score_ix, sign_multiplier, filter_pases)
    
    print("Done.")
