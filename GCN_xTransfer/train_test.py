import os
import numpy as np
import pandas as pd
import math
from configuration import printt


class TrainTest:
    def __init__(self, results_processor=None):
        self.results_processor = results_processor

    def fit_model(self, exp_specs, data, model):
        """
        trains model by iterating minibatches for specified number of epochs
        """
        printt("Fitting Model")
        # train for specified number of epochs
        for epoch in range(1, exp_specs["num_epochs"] + 1):
            self.train_epoch(data["train"], model, exp_specs["minibatch_size"])
            printt("Epoch %d" % epoch)

        # calculate train and test metrics
        headers, result = self.results_processor.process_results(exp_specs, data, model, "epoch_" + str(epoch))
        
        # clean up
        self.results_processor.reset()
        model.close()
        return headers, result

    def fit_model_and_activate(self, exp_specs, data, model, layer_specs, rep, outdir):
        printt("Fitting Model")
        # train for specified number of epochs
        for epoch in range(1, exp_specs["num_epochs"] + 1):
            self.train_epoch(data["train"], model, exp_specs["minibatch_size"])
            # printt("Epoch %d" % epoch)

            if(epoch % 10 == 0):
                # calculate train and test metrics
                headers, result = self.results_processor.process_results(exp_specs, data, model, "epoch_" + str(epoch))
                if(epoch==10):
                # create results log
                    results_log = os.path.join(outdir, "path_"+str(rep)+".csv")
                    with open(results_log, 'w') as f:
                        f.write("{}\n".format(",".join(["epoch"] + headers)))
                
                # write results to file
                with open(results_log, 'a') as f:
                    f.write("{}, {}\n".format(epoch, ",".join([str(r) for r in result])))
            
                self.results_processor.reset()
                

        # calculate train and test metrics
        headers, result = self.results_processor.process_results(exp_specs, data, model, "epoch_" + str(epoch))

        printt("*"*30)
        printt("Activations for validation set")
        dir_out = outdir + "/val_"+str(rep)+"/"
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        self.activate_for_proteins(data["val"], model, layer_specs, dir_out)

        printt("*"*30)
        printt("Activations for test set")
        dir_out = outdir + "/test_"+str(rep)+"/"
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        self.activate_for_proteins(data["test"], model, layer_specs, dir_out)

        printt("*"*30)
        printt("Saving network weights")
        dir_out = outdir + "/weights_"+str(rep)+"/"
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        self.extract_weights(data["test"], model, layer_specs, dir_out)

        # clean up
        self.results_processor.reset()
        model.close()
        return headers, result

    def activate_for_proteins(self, data, model, layer_specs, outdir):
        num_proteins = len(data)
        for protein in range(num_proteins):
            prot_data = data[protein]
            prot_name = prot_data["complex_code"]
            #print("Evaluating protein: %s" % prot_name)
            
            dir_prot = outdir + str(protein) + "." + prot_name + "/"
            if not os.path.exists(dir_prot):
                os.mkdir(dir_prot)
            list_activations = model.activate(prot_data)

            for layer_idx in range(len(layer_specs)):
                layer_name = str(layer_idx) +"."+ layer_specs[layer_idx][0]
                dir_layer = dir_prot + layer_name + "/"
                if not os.path.exists(dir_layer):
                    os.mkdir(dir_layer)
                layer_activations = list_activations[layer_idx]
                for out_layer in layer_activations.keys():
                    out_activations = pd.DataFrame(layer_activations[out_layer],index=None,columns=None)
                    out_activations.to_csv(dir_layer+out_layer+".csv",header=False,index=False)

    def extract_weights(self, data, model, layer_specs, outdir):
            list_weights = model.weights(data[0])

            for layer_idx in range(len(layer_specs)):
                layer_name = str(layer_idx) +"."+ layer_specs[layer_idx][0]
                dir_layer = outdir + layer_name + "/"
                if not os.path.exists(dir_layer):
                    os.mkdir(dir_layer)
                layer_weights = list_weights[layer_idx]
                for out_layer in layer_weights.keys():
                    weight_mat = layer_weights[out_layer]
                    out_weights = pd.DataFrame(weight_mat,index=None,columns=None)
                    out_weights.to_csv(dir_layer+out_layer+".csv",header=False,index=False)

    def train_epoch(self, data, model, minibatch_size):
        """
        Trains model for one pass through training data, one protein at a time
        Each protein is split into minibatches of paired examples.
        Features for the entire protein is passed to model, but only a minibatch of examples are passed
        """
        prot_perm = np.random.permutation(len(data))
        # loop through each protein
        prot_idx = 0
        for protein in prot_perm:
            # extract just data for this protein
            prot_idx += 1
            
            prot_data = data[protein]
            pair_examples = prot_data["label"]
            n = pair_examples.shape[0]
            shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
            num_resid_r = prot_data["label_r"].shape[0]

            num_batches = int(math.ceil(n*1.0 / minibatch_size))
            
            # loop through each minibatch
            for i in range(num_batches):
                # extract data for this minibatch
                index = int(i * minibatch_size)
                examples = pair_examples[shuffle_indices[index: index + minibatch_size],:]
                m_batch = examples.shape[0]
                m_batch = n

                idx_repeat_l = np.reshape(np.tile(np.reshape(np.arange(m_batch),[-1,1]),num_resid_r),[-1,1])
                idx_repeat_r = np.reshape(np.tile(np.arange(num_resid_r),m_batch),[-1,1])
                idx_pairs_l = np.concatenate((idx_repeat_l,idx_repeat_r),axis=1)

                minibatch = {}
                for feature_type in prot_data:
                    if feature_type == "label":
                        minibatch["label"] = examples
                    elif feature_type == "pairs":
                        minibatch["pairs"] = idx_pairs_l
                    else:
                        minibatch[feature_type] = prot_data[feature_type]
                    
                # train the model
                model.train(minibatch)