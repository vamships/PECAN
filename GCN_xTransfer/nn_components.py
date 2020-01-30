import numpy as np
import tensorflow as tf

__all__ = [
    "ann",
    "node_average",
    "attend_on_other",
    "attend_on_other_recurrent",
    "average_predictions",
    "initializer",
    "nonlinearity",
]

""" ====== Layers ====== """
""" All layers have as first two parameters:
        - input: input tensor or tuple of input tensors
        - params: dictionary of parameters, could be None
"""

def ann(input, params, params_preTrain=None, out_dims=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    input = tf.nn.dropout(input, dropout_keep_prob)
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    if params_preTrain is None:
        if params is None:
            W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
            b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
            params = {"W": W, "b": b}
        else:
            W, b = params["W"], params["b"]
    else:
        W = tf.Variable(params_preTrain["W"],name="w",trainable=trainable)
        b = tf.Variable(params_preTrain["b"],name="b",trainable=trainable)
        params = {"W": W, "b": b}
    
    Z = tf.matmul(input, W) + b

    regularizer = tf.constant(0.0)
    
    return Z, params, regularizer

def node_average(input, params, params_preTrain=None, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    #print("In node_average: %d" % (v_shape[1].value))
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value
    if params_preTrain is None:
        if params is None:
            # create new weights
            Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc", trainable=trainable)  # (v_dims, filters)
            Wn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn", trainable=trainable)  # (v_dims, filters)
            b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
            params = {"Wn": Wn, "Wc": Wc, "b": b}
        else:
            Wn, Wc = params["Wn"], params["Wc"]
            filters = Wc.get_shape()[-1].value
            b = params["b"]
    else:
        Wc = tf.Variable(params_preTrain["Wc"], name="Wc", trainable=trainable)  # (v_dims, filters)
        Wn = tf.Variable(params_preTrain["Wn"], name="Wn", trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(params_preTrain["b"], name="b", trainable=trainable)
        params = {"Wn": Wn, "Wc": Wc, "b": b}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    v_Wn = tf.matmul(vertices, Wn, name="v_Wn")  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wn, nh_indices), 1),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)

    nonlin = nonlinearity("relu")
    sig = Zn + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params

def attend_on_other(input, params, params_preTrain=None, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    input1, input2, pairs = input

    num_resid_l = tf.reduce_max(pairs[:,0])+1
    num_resid_r = tf.reduce_max(pairs[:,1])+1

    in_shape = input1.get_shape()
    rec_shape = input2.get_shape()
    if params_preTrain is None:
        if params is None:
            # create new weights
            Wc = tf.Variable(initializer("he", (in_shape[1].value, in_shape[1].value)), name="Wc", trainable=trainable)  # (v_dims, filters)
            params = {"Wc": Wc}
        else:
            Wc = params["Wc"]
            filters = Wc.get_shape()[-1].value
    else:
        Wc = tf.Variable(params_preTrain["Wc"], name="Wc", trainable=trainable)
        params = {"Wc": Wc}

    # generate vertex signals
    Zc_intermediate_pre = tf.matmul(input1, Wc, name="Zc_temp")  # (n_verts, filters)
    Zc_intermediate_post = tf.matmul(tf.transpose(Wc),tf.transpose(input2), name="Zc_temp")
    sig = tf.matmul(Zc_intermediate_pre, Zc_intermediate_post, name="sig")

    h = tf.nn.relu(sig)

    h_flat = tf.reshape(h, [-1])
    h_flat_norm = tf.nn.l2_normalize(h_flat)
    h_norm = tf.reshape(h_flat_norm, [num_resid_l, -1])

    h_softmax = h_norm

    feats_other = tf.transpose(tf.matmul(input2,h_softmax,transpose_a=True,transpose_b=True))

    return feats_other, h, h_softmax, params

def attend_on_other_recurrent(input, params, params_preTrain=None, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    input1, input2, pairs, preds_semi = input

    num_resid_l = tf.reduce_max(pairs[:,0])+1
    num_resid_r = tf.reduce_max(pairs[:,1])+1

    in_shape = input1.get_shape()
    rec_shape = input2.get_shape()

    if params_preTrain is None:
        if params is None:
            # create new weights
            Wc = tf.Variable(initializer("he", (in_shape[1].value, in_shape[1].value)), name="Wc", trainable=trainable)  # (v_dims, filters)
            params = {"Wc": Wc}
        else:
            Wc = params["Wc"]
            filters = Wc.get_shape()[-1].value
    else:
        Wc = tf.Variable(params_preTrain["Wc"], name="Wc", trainable=trainable)
        params = {"Wc": Wc}

    # generate vertex signals
    Zc_intermediate_pre = tf.matmul(input1, Wc, name="Zc_temp")  # (n_verts, filters)
    Zc_intermediate_post = tf.matmul(tf.transpose(Wc),tf.transpose(input2), name="Zc_temp")
    sig = tf.matmul(Zc_intermediate_pre, Zc_intermediate_post, name="sig")

    h = tf.nn.relu(sig)

    h_scaled = h * preds_semi[:,0]

    h_flat = tf.reshape(h_scaled, [-1, 1])

    h_flat_norm = tf.nn.l2_normalize(h_flat)
    h_norm = tf.reshape(h_flat_norm, [num_resid_l, -1])

    h_softmax = h_norm
    #h_softmax = labels_attn

    feats_other = tf.transpose(tf.matmul(input2,h_softmax,transpose_a=True,transpose_b=True))

    return feats_other, h_scaled, h_softmax, params


def average_predictions(input, _, **kwargs):
    combined = tf.reduce_mean(tf.stack(tf.split(input, 2)), 0)
    return combined, None


""" ======== Non Layers ========= """


def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)


def nonlinearity(nl):
    if nl == "relu":
        return tf.nn.relu
    elif nl == "leaky_relu":
        return tf.nn.leaky_relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x
