import os
import sys
import yaml
import traceback
import cPickle
import tensorflow as tf
import numpy as np
import pandas as pd
from configuration import mode_experiment, data_directory, experiment_directory, output_directory, printt
from train_test import TrainTest
from sample_pw_classifier_attn2 import PWClassifier
from results_processor import ResultsProcessor

## Random Seeds
# each random seed represents an experimental replication.
# You can add or remove list elements to change the number
# of replications for an experiment.
seeds = [
    {"tf_seed": 649737, "np_seed": 29820},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 252356, "np_seed": 703889},
    {"tf_seed": 343053, "np_seed": 999360},
    {"tf_seed": 743746, "np_seed": 67440}
]

# Load experiment specified in system args
exp_file = "node_edge_attn2.yml"
printt("Running Experiment File: {}".format(exp_file))
f_name = exp_file.split(".")[0] if "." in exp_file else exp_file
exp_specs = yaml.load(open(os.path.join(experiment_directory, exp_file), 'r').read())

# setup output directory
outdir = os.path.join(output_directory, f_name)
if not os.path.exists(outdir):
    os.mkdir(outdir)
results_processor = ResultsProcessor()

# create results log
results_log = os.path.join(outdir, "results.csv")
with open(results_log, 'w') as f:
    f.write("")

# write experiment specifications to file
with open(os.path.join(outdir, "experiment.yml"), 'w') as f:
    f.write("{}\n".format(yaml.dump(exp_specs)))

# pre-process to include residue indices
def preProcessData(data):
    num_complex = len(data)
    for idx_complex in range(num_complex):
        data_complex = data[idx_complex]

        num_resid_r = data_complex["r_vertex"].shape[0]
        num_resid_l = data_complex["l_vertex"].shape[0]

        idx_repeat_l = np.reshape(np.tile(np.reshape(np.arange(num_resid_l),[-1,1]),num_resid_r),[-1,1])
        idx_repeat_r = np.reshape(np.tile(np.arange(num_resid_r),num_resid_l),[-1,1])
        idx_pairs_l = np.concatenate((idx_repeat_l,idx_repeat_r),axis=1)
        data[idx_complex]["pairs"] = idx_pairs_l
        
        # Use only up to 15 neighbors during convolution
        data[idx_complex]["l_hood_indices"] = data[idx_complex]["l_hood_indices"][:,:15,:]
        data[idx_complex]["r_hood_indices"] = data[idx_complex]["r_hood_indices"][:,:15,:]

        data[idx_complex]["l_edge"] = data[idx_complex]["l_edge"][:,:15,:]
        data[idx_complex]["r_edge"] = data[idx_complex]["r_edge"][:,:15,:]

# Remove antigens as primary protein
def remove_ags(data):
    printt("Removing Ags")
    new_data = []
    for data_idx in range(len(data)):
        if data_idx % 2 == 1:
            new_data.append(data[data_idx])
    return new_data

# Remove antibodies as primary protein
def remove_abs(data):
    printt("Removing Abs")
    new_data = []
    for data_idx in range(len(data)):
        if data_idx % 2 == 0:
            new_data.append(data[data_idx])
    return new_data

# Get pretraining weights
dir_preTrain = "../results_final/results_protein_base/"+f_name+"/weights_0/"

wt_conv_center = np.genfromtxt(dir_preTrain+"0.node_average/wt_conv_center.csv",dtype=np.float32,delimiter=",")
wt_conv_neighb = np.genfromtxt(dir_preTrain+"0.node_average/wt_conv_neighb.csv",dtype=np.float32,delimiter=",")
bs_conv = np.genfromtxt(dir_preTrain+"0.node_average/bias_conv.csv",dtype=np.float32,delimiter=",")
bs_conv = np.transpose(np.expand_dims(bs_conv,axis=1))
params_conv = {"Wc":wt_conv_center, "Wn":wt_conv_neighb, "b":bs_conv}

wt_conv2_center = np.genfromtxt(dir_preTrain+"1.node_average/wt_conv_center.csv",dtype=np.float32,delimiter=",")
wt_conv2_neighb = np.genfromtxt(dir_preTrain+"1.node_average/wt_conv_neighb.csv",dtype=np.float32,delimiter=",")
bs_conv2 = np.genfromtxt(dir_preTrain+"1.node_average/bias_conv.csv",dtype=np.float32,delimiter=",")
bs_conv2 = np.transpose(np.expand_dims(bs_conv2,axis=1))
params_conv2 = {"Wc":wt_conv2_center, "Wn":wt_conv2_neighb, "b":bs_conv2}

wt_attn = np.genfromtxt(dir_preTrain+"2.attend_on_other/wt_attention.csv",dtype=np.float32,delimiter=",")
params_attn = {"Wc":wt_attn}

wt_dense = np.genfromtxt(dir_preTrain+"3.ann/wt_dense.csv",dtype=np.float32,delimiter=",")
wt_dense = np.expand_dims(wt_dense,axis=1)
bs_dense = np.array([np.genfromtxt(dir_preTrain+"3.ann/bias_dense.csv",dtype=np.float32,delimiter=",")])
params_dense = {"W": wt_dense, "b": bs_dense}

print(wt_conv2_center.shape,wt_conv2_neighb.shape, bs_conv2.shape)
params_preTrain = [params_conv, params_conv2, params_attn, params_dense]

first_experiment = True
for name, experiment in exp_specs["experiments"]:
    train_data_file = os.path.join(data_directory, experiment["train_data_file"])
    val_data_file = os.path.join(data_directory, experiment["val_data_file"])
    test_data_file = os.path.join(data_directory, experiment["test_data_file"])
    try:
        print("*" * 30)
        printt("{} network".format(name))
        printt("Loading train data")
        train_data = cPickle.load(open(train_data_file))
        preProcessData(train_data)

        printt("Loading validation data")
        val_data = cPickle.load(open(val_data_file))
        preProcessData(val_data)

        printt("Loading test data")
        test_data = cPickle.load(open(test_data_file))
        preProcessData(test_data)

        # Prepare data as per task
        if mode_experiment=="epitope":
            train_data = remove_abs(train_data)
            val_data = remove_abs(val_data)
            test_data = remove_abs(test_data)
        else:
            train_data = remove_ags(train_data)
            val_data = remove_ags(val_data)
            test_data = remove_ags(test_data)

        print(len(train_data), len(val_data), len(test_data))

        data = {"train": train_data, "val": val_data, "test": test_data}
        for i, seed_pair in enumerate(seeds):
            printt("{}: rep{}".format(name, i))
            # set tensorflow and numpy seeds
            tf.set_random_seed(seed_pair["tf_seed"])
            np.random.seed(int(seed_pair["np_seed"]))
            
            # build model
            printt("Building model")
            learning_rate = 0.001
            model = PWClassifier(experiment["layers"], experiment["layer_args"], data["train"], learning_rate, params_preTrain, 0.1, outdir)
            
            # train and test the model
            headers, results = TrainTest(results_processor).fit_model_and_activate(exp_specs, data, model, experiment["layers"], i, outdir)
            
            # write headers to file if haven't already
            if first_experiment:
                with open(results_log, 'a') as f:
                    f.write("{}\n".format(",".join(["file", "experiment", "rep"] + headers)))
                first_experiment = False
            
            # write results to file
            with open(results_log, 'a') as f:
                f.write("{}, {}, {}, {}\n".format(f_name, name, i, ",".join([str(r) for r in results])))
            print("*" * 30)

    except Exception as er:
        if er is KeyboardInterrupt:
            raise er
        ex_str = traceback.format_exc()
        printt(ex_str)
        printt("Experiment failed: {}".format(exp_specs))
