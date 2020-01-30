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
        
        self.wt_dense = None
        self.output_dense = None
        self.regularizer = None
        self.output_avg = None

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
            # Prediction Layer
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

            input1 = tf.gather(self.in_vertex1, self.examples)

            name = "{}_{}".format(type, i)
            with tf.name_scope(name):
                self.output_dense, params, self.regularizer = layer_fn(input1, None, params_preTrain, **args)

                self.wt_dense = params["W"]

                if params is not None and len(params.items()) > 0:
                    self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})

            self.preds = self.output_dense

            #---------------------------------------
            # Loss and optimizer
            #---------------------------------------

            with tf.name_scope("loss"):
                scale_vector = (pn_ratio * (self.labels - 1) / -2) + ((self.labels + 1) / 2)
                
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
        return self.run_graph([self.train_op, self.loss], data, "train")

    def weights(self, data):
        wt_dense = self.run_graph(self.wt_dense, data, "test")
        return [{"wt_dense":wt_dense}]

    def activate(self, data):
        output_dense, output_preds, input_labels = self.run_graph([self.output_dense, self.preds, self.labels], data, "test")
        return [{"output_dense":output_dense,
                "output_preds":output_preds, "labels":input_labels}]
    
    def get_nodes(self):
        return [n for n in self.graph.as_graph_def().node]

    def close(self):
        with self.graph.as_default():
            self.sess.close()