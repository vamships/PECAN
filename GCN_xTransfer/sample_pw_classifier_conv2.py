import os
import copy

import numpy as np
import tensorflow as tf

import nn_components

__all__ = [
    "PWClassifier",
]


class PWClassifier(object):
    def __init__(self, layer_specs, layer_args, train_data, learning_rate, params_preTrain, pn_ratio, outdir):
        """ Assumes same dims and nhoods for l_ and r_ """
        self.layer_args = layer_args
        self.params = {}
        # tf stuff:
        self.graph = tf.Graph()
        self.sess = None
        self.preds = None
        self.labels = None
        
        self.output_conv = None
        self.wt_conv_center = None
        self.wt_conv_neighb = None

        self.output_conv2 = None
        self.wt_conv2_center = None
        self.wt_conv2_neighb = None

        self.wt_dense = None
        self.output_dense = None
        self.regularizer = None
        #self.scale_vector = None

        #################################################################
        # get details of data
        self.in_nv_dims = train_data[0]["l_vertex"].shape[-1]
        #print("Vertices: %d" % self.in_nv_dims)
        self.in_ne_dims = train_data[0]["l_edge"].shape[-1]
        #print("Edges: %s" % self.in_ne_dims)
        self.in_nhood_size = train_data[0]["l_hood_indices"].shape[1]
        #print("Neighborhood size: %d" % self.in_nhood_size)
        with self.graph.as_default():
            # shapes and tf variables
            self.in_vertex1 = tf.placeholder(tf.float32, [None, self.in_nv_dims], "vertex1")
            self.in_vertex2 = tf.placeholder(tf.float32, [None, self.in_nv_dims], "vertex2")
            self.in_edge1 = tf.placeholder(tf.float32, [None, self.in_nhood_size, self.in_ne_dims], "edge1")
            self.in_edge2 = tf.placeholder(tf.float32, [None, self.in_nhood_size, self.in_ne_dims], "edge2")
            self.in_hood_indices1 = tf.placeholder(tf.int32, [None, self.in_nhood_size, 1], "hood_indices1")
            self.in_hood_indices2 = tf.placeholder(tf.int32, [None, self.in_nhood_size, 1], "hood_indices2")
            
            input1 = self.in_vertex1, self.in_edge1, self.in_hood_indices1
            input2 = self.in_vertex2, self.in_edge2, self.in_hood_indices2
            
            self.examples = tf.placeholder(tf.int32, [None], "examples")
            self.labels = tf.placeholder(tf.float32, [None], "labels")
            self.scale_vector = tf.placeholder(tf.float32, [None], "scale_vector")
            self.train_phase = tf.placeholder(tf.bool, name="training")
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")
            
            #---------------------------------------
            # Convolution Layer #1
            #---------------------------------------

            i = 0

            layer = layer_specs[i]

            args = copy.deepcopy(layer_args)
            args["dropout_keep_prob"] = self.dropout_keep_prob

            type = layer[0]

            print("Assembling layer %d : %s with dropout %f" % (i,layer[0], self.layer_args["dropout_keep_prob"]))
            # number of neurons
            args2 = layer[1] if len(layer) > 1 else {}
            # merge flag
            flags = layer[2] if len(layer) > 2 else None    
            args.update(args2)  # local layer args override global layer args
            # get empty layer 
            layer_fn = getattr(nn_components, type)

            name = "leg1_{}_{}".format(type, i)
            #print("Leg1: ",name)
            with tf.name_scope(name):
                # initialize layer with parameters
                self.output_conv, params = layer_fn(input1, None, params_preTrain[i], **args)
                if params is not None:
                    self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})

                self.wt_conv_center = params["Wc"]
                self.wt_conv_neighb = params["Wn"]

            #---------------------------------------
            # Convolution Layer #2
            #---------------------------------------

            i += 1

            layer = layer_specs[i]

            args = copy.deepcopy(layer_args)
            args["dropout_keep_prob"] = self.dropout_keep_prob

            type = layer[0]

            print("Assembling layer %d : %s with dropout %f" % (i,layer[0], self.layer_args["dropout_keep_prob"]))
            # number of neurons
            args2 = layer[1] if len(layer) > 1 else {}
            # merge flag
            flags = layer[2] if len(layer) > 2 else None    
            args.update(args2)  # local layer args override global layer args
            # get empty layer 
            layer_fn = getattr(nn_components, type)

            # self.output_conv = tf.nn.dropout(self.output_conv, self.dropout_keep_prob)
            input1 = self.output_conv, self.in_edge1, self.in_hood_indices1

            name = "leg1_{}_{}".format(type, i)
            #print("Leg1: ",name)
            with tf.name_scope(name):
                # initialize layer with parameters
                self.output_conv2, params = layer_fn(input1, None, params_preTrain[i], **args)
                if params is not None:
                    self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})

                self.wt_conv2_center = params["Wc"]
                self.wt_conv2_neighb = params["Wn"]


            #---------------------------------------
            # Prediction Layer
            #---------------------------------------

            i += 1

            layer = layer_specs[i]
            args = copy.deepcopy(layer_args)
            args["dropout_keep_prob"] = self.dropout_keep_prob
            type = layer[0]
            print("Assembling layer %d : %s with dropout %f" % (i,layer[0], self.layer_args["dropout_keep_prob"]))

            # number of neurons
            args2 = layer[1] if len(layer) > 1 else {}
            # merge flag
            flags = layer[2] if len(layer) > 2 else None                
            args.update(args2)  # local layer args override global layer args
            # get empty layer 
            layer_fn = getattr(nn_components, type)

            sel_conv2 = tf.gather(self.output_conv2, self.examples)

            input1 = sel_conv2
            
            name = "{}_{}".format(type, i)
            with tf.name_scope(name):
                self.output_dense, params, self.regularizer = layer_fn(input1, None, params_preTrain[i], **args)

                self.wt_dense = params["W"]

                if params is not None and len(params.items()) > 0:
                    self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})

            self.preds = self.output_dense

            #---------------------------------------
            # Loss and optimizer
            #---------------------------------------

            with tf.name_scope("loss"):
                #scale_vector = self.scale_vector
                scale_vector = (0.1 * (self.labels - 1) / -2) + ((self.labels + 1) / 2)
                # label predictions
                logits = tf.concat([-self.preds, self.preds], axis=1)

                # one hot labels
                labels = tf.stack([(self.labels - 1) / -2, (self.labels + 1) / 2], axis=1) 
                
                self.loss = tf.losses.softmax_cross_entropy(labels, logits, weights=scale_vector)

            with tf.name_scope("optimizer"):
                self.train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(self.loss)

            # set up tensorflow session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def run_graph(self, outputs, data, tt, options=None, run_metadata=None):
        with self.graph.as_default():
            dropout_keep = 1.0
            is_training = False
            if tt == "train":
                is_training = True
            if tt == "train" and "dropout_keep_prob" in self.layer_args:
                dropout_keep = self.layer_args["dropout_keep_prob"]
            feed_dict = {
                self.in_vertex1: data["l_vertex"], self.in_edge1: data["l_edge"],
                self.in_vertex2: data["r_vertex"], self.in_edge2: data["r_edge"],
                self.in_hood_indices1: data["l_hood_indices"],
                self.in_hood_indices2: data["r_hood_indices"],
                self.examples: data["label"][:, 0],
                self.labels: data["label"][:, 1],
                self.dropout_keep_prob: dropout_keep,
                self.train_phase: is_training}
            return self.sess.run(outputs, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def get_labels(self, data):
        return {"label": data["label"][:, 1, np.newaxis]}

    def predict(self, data):
        results = self.run_graph([self.loss, self.preds], data, "test")
        results = {"label": results[1], "loss": results[0]}
        return results

    def loss(self, data):
        return self.run_graph(self.loss, data, "test")

    def train(self, data):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self.run_graph([self.train_op, self.loss], data, "train")

    def weights(self, data):
        output_list = self.run_graph([self.wt_conv_center, self.wt_conv_neighb,
                                        self.wt_conv2_center, self.wt_conv2_neighb,
                                        self.wt_dense], data, "test")
        return [{"wt_conv_center":output_list[0], "wt_conv_neighb":output_list[1]},
                {"wt_conv_center":output_list[2], "wt_conv_neighb":output_list[3]},
                {"wt_dense":output_list[4]}]
    
    def activate(self, data): 
        output_list = self.run_graph([self.output_conv,
                                        self.output_conv2,
                                        self.output_dense,
                                        self.preds,
                                        self.labels], data, "test")
        return [{"output_conv":output_list[0]},
                {"output_conv":output_list[1]},
                {"output_dense":output_list[2], "output_preds":output_list[3], "labels":output_list[4]}]

    def get_nodes(self):
        return [n for n in self.graph.as_graph_def().node]

    def close(self):
        with self.graph.as_default():
            self.sess.close()