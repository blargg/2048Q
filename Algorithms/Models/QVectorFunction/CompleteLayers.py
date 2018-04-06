import tensorflow as tf


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
                 trainable=True):
        self.state = tf.placeholder(tf.float32, shape=[None] + stateDims)
        self.state_flat = tf.layers.flatten(self.state)

        self.layer1 = self.denseLayer("layer1", 10, trainable=trainable)
        self.tensor1 = self.layer1.apply(self.state_flat)

        self.layer2 = self.denseLayer("layer2", 10, trainable=trainable)
        self.tensor2 = self.layer2.apply(self.tensor1)

        self.action_layer = self.finalLayer("action_layer",
                                            numActions,
                                            trainable=trainable)
        self.actions = self.action_layer.apply(self.tensor2)

    def denseLayer(self, name, num_output=10, trainable=True):
        init = tf.initializers.random_normal()
        return tf.layers.Dense(units=num_output,
                               activation=tf.nn.relu,
                               kernel_initializer=init,
                               bias_initializer=init,
                               name=name)

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
        return [self.layer1.kernel,
                self.layer1.bias,
                self.layer2.kernel,
                self.layer2.bias,
                self.action_layer.kernel,
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
