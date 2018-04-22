import tensorflow as tf
from util.tensorflow.DenseLayers import DenseLayers


class CompleteLayers:
    """Constructs the q-function model for QNet
    Network consists of 3 layers, were every output in the pervious layer
    connects to the next layer.

    Notable tensors
    state: The input placeholder. Has shape [none, stateDims] for batching.
           In q-learning this represents the current state task.
    actions: The q-values calculated. Has shape [none, numActions]
             Each index represents the q-value for taking the action numbered
             by that index.
    """
    def __init__(self,
                 stateDims,
                 numActions,
                 trainable=True,
                 keep_prob=None):
        # Variables
        if keep_prob is None:
            self.keep_prob =\
                tf.get_variable("keep_prob",
                                initializer=tf.constant(1.0),
                                trainable=False)
        else:
            self.keep_prob = keep_prob

        # TODO decompose parts
        self.state = tf.placeholder(tf.int32, shape=[None] + stateDims)
        self.encoded = tf.one_hot(self.state, 10)
        self.dropout_state = tf.nn.dropout(self.encoded, self.keep_prob)
        self.state_flat = tf.layers.flatten(self.dropout_state)

        layerSizes = [10, 10]
        self.denseLayers = DenseLayers(layerSizes,
                                       trainable=trainable,
                                       keep_prob=self.keep_prob)
        lastLayer = self.denseLayers.apply(self.state_flat)

        self.action_layer = self.finalLayer("action_layer",
                                            numActions,
                                            trainable=trainable)
        self.actions = self.action_layer.apply(lastLayer)

    def finalLayer(self, name, num_output, trainable=True):
        init = tf.initializers.random_normal()
        return tf.layers.Dense(units=num_output,
                               kernel_initializer=init,
                               bias_initializer=init,
                               name=name)

    def get_all_variables(self):
        """Returns all variables used by this part of the model in an
        interable. The variables are returned in a consistant order, such that
        copying the variable to another version of this object in order will
        produce the same function"""
        return self.denseLayers.allVariables() +\
            [self.action_layer.kernel,
             self.action_layer.bias]


def mkConstructor(inputDims, outputDims):
    """Helper to make a constructor function whose arguments match qnet.
    Returns function of type (Trainable:Bool) -> CompleteLayers"""
    def construct(trainable=True):
        """Constructs a CompleteLayers Graph component in tensorflow.
        All variables will be trainable based on the argument to this
        function"""
        return CompleteLayers(inputDims, outputDims)

    return construct
