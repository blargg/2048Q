import tensorflow as tf

from Algorithms.Models.QVectorFunction import OneHotBoard
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
        if not trainable:
            # if not training, then don't use dropout
            self.keep_prob = None
        elif keep_prob is None:
            self.keep_prob =\
                tf.get_variable("keep_prob",
                                initializer=tf.constant(1.0),
                                trainable=False)
        else:
            self.keep_prob = keep_prob

        self.boardPlaceholder = OneHotBoard()
        self.state = self.boardPlaceholder.input_state

        layerSizes = [10, 10]
        self.denseLayers = DenseLayers(layerSizes,
                                       trainable=trainable,
                                       keep_prob=self.keep_prob)
        lastLayer = self.denseLayers \
            .apply(self.boardPlaceholder.state_representation)

        self.action_layer = self.finalLayer("action_layer",
                                            numActions,
                                            trainable=trainable)
        self.actions = self.action_layer.apply(lastLayer)

    def finalLayer(self, name, num_output, trainable=True):
        init = tf.initializers.random_normal()
        return tf.layers.Dense(units=num_output,
                               kernel_initializer=init,
                               bias_initializer=init,
                               trainable=trainable,
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
        return CompleteLayers(inputDims, outputDims,
                              trainable=trainable)

    return construct
