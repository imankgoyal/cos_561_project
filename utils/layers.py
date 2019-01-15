import tensorflow as tf

from tensorflow.python.training import moving_averages

# Batch_norm adapted from: 
#   https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/ops.py 
#   https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py
#   https://stackoverflow.com/questions/41819080/how-do-i-use-batch-normalization-in-a-multi-gpu-setting-in-tensorflow
# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'
def batch_norm(inp,
               is_training,
               center=True,
               scale=True,
               epsilon=0.001,
               decay=0.99,
               name=None,
               reuse=None):

    """Adds a Batch Normalization layer.
    
    Args:
        inp: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
        is_training: whether or not the model is in training mode.
        center: If True, subtract beta. If False, beta is not created and
            ignored.
        scale: If True, multiply by gamma. If False, gamma is
            not used. When the next layer is linear (also e.g. ReLU), this can be
            disabled since the scaling can be done by the next layer.
        epsilon: small float added to variance to avoid dividing by zero.
        decay: decay for the moving average.
        name: Optional scope for variable_scope.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
    
    Returns:
        a tensor representing the output of the operation.
    """

    if name == None:
        name = "batch_norm"

    inputs_shape = inp.get_shape()
    with tf.variable_scope(name, reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable(
                'beta',
                shape=params_shape,
                initializer=tf.zeros_initializer(),
                trainable=True)
        if scale:
            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                initializer=tf.ones_initializer(),
                trainable=True)

        moving_mean = tf.get_variable(
            'moving_mean',
            params_shape,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance',
            params_shape,
            initializer=tf.ones_initializer(),
            trainable=False)
        
        def mean_var_from_data():
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inp, axis)
            return mean, variance

        mean, variance = tf.cond(
                        pred=is_training,
                        true_fn=mean_var_from_data,
                        false_fn=lambda: (moving_mean, moving_variance))
        
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, decay)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance) 

    # Normalize the activations.
    outputs = tf.nn.batch_normalization(inp, mean, variance, beta, gamma,
                                        epsilon)

    return outputs


def residual_fc(
    inp,
    is_training,
    relu_after=True,
    add_bn=True, 
    name=None,
    reuse=None):
    
    """ Returns a residual block fc layer """
    if name == None:
        name = "residual_fc"

    inp_dim = int(inp.shape[-1])
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=inp_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)
        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm_1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=inp_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)
        
        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm_2",
                reuse=reuse)

        if relu_after:
            out = tf.nn.relu(inp + out2)
        else:
            out = inp + out2

    return out


def proj_residual_fc(
    inp,
    is_training,
    out_dim,
    relu_after=True,
    add_bn=True, 
    name=None,
    reuse=None):

    """ Returns a residual block fc layer with projection """
    if name == None:
        name = "proj_residual_fc"

    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)
        if add_bn:
            out1 = batch_norm(
                inp=out1,
                is_training=is_training,
                name="norm_1",
                reuse=reuse)

        out1 = tf.nn.relu(out1)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)
        
        if relu_after and add_bn:
            out2 = batch_norm(
                inp=out2,
                is_training=is_training,
                name="norm_2",
                reuse=reuse)

        out3 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_3", reuse=reuse)

        if relu_after:
            out = tf.nn.relu(out3 + out2)
        else:
            out = out3 + out2

    return out


def ser_fc_layers(
    inp,
    is_training,
    num_hid,
    hid_dim,
    out_dim, 
    relu_after=False,
    add_bn=True,
    name=None,
    reuse=None):
    """ Returns a series of fully connected layer
        
        Here, each hidden layer means a residual block. The first fc layer maps 
        the inp to hid_dim. Then, a series of (num_hid-1) fc layers. Finally, a 
        fc layer which maps to out_dim.  There is a relu in all the hidden
        dimension. The presense of the last activation is controlled using
        relu_after. If add_bn, a bn layer added before every relu
    """

    if name == None:
        name = "ser_fc_layer"

    with tf.variable_scope(name, reuse=reuse):
        if num_hid > 0:
            if int(inp.shape[-1]) == hid_dim:
                inp = residual_fc(
                    inp=inp,
                    is_training=is_training,
                    relu_after=True,
                    add_bn=add_bn, 
                    name="layer_0",
                    reuse=reuse)
            else:
                inp = proj_residual_fc(
                    inp=inp,
                    is_training=is_training,
                    out_dim=hid_dim,
                    relu_after=True,
                    add_bn=add_bn, 
                    name="layer_0",
                    reuse=reuse)

            for i in range(num_hid - 1):
                inp = residual_fc(
                    inp=inp,
                    is_training=is_training,
                    relu_after=True,
                    add_bn=add_bn, 
                    name="layer_{}".format(i + 1),
                    reuse=reuse)

        if hid_dim == out_dim:
            out = residual_fc(
                inp=inp,
                is_training=is_training,
                relu_after=relu_after,
                add_bn=add_bn, 
                name="layer_last",
                reuse=reuse)
        else:
            out = proj_residual_fc(
                inp=inp,
                is_training=is_training,
                out_dim=out_dim,
                relu_after=relu_after,
                add_bn=add_bn, 
                name="layer_last",
                reuse=reuse)

    return out

def final_fc_layers(
    inp,
    out_dim,
    name=None,
    reuse=None):
    """ These are to be used before predicting the logits. As Hei suggested,
        the fc layers of the last layer should not have bn so no bn capability.  
    """

    if name == None:
        name = "final_fc_layers"

    inp_dim = int(inp.shape[-1])
    with tf.variable_scope(name, reuse=reuse):
        out1 = tf.contrib.layers.fully_connected(
                    inp,
                    num_outputs=inp_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_1", reuse=reuse)

        out2 = tf.contrib.layers.fully_connected(
                    out1,
                    num_outputs=out_dim,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope="layer_2", reuse=reuse)

    return out2