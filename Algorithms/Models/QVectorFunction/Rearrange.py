import tensorflow as tf
from util.tensorflow.DenseLayers import DenseLayers


class Rearrange:
    def __init__(self,
                 stateDims,
                 numActions,
                 trainable=True):
        self.state = tf.placeholder(tf.float32, shape=[None] + stateDims)
        self.state_flat = tf.layers.flatten(self.state)
        if trainable:
            self.keep_prob =\
                tf.get_variable("keep_prob",
                                initializer=tf.constant(1.0),
                                trainable=False)
        else:
            self.keep_prob = None

        [_, stateVectorSize] = self.state_flat.get_shape()
        # tensor that can potentially represent a rearrangement of the input
        self.rearrangeLayers = []
        self.arrangementTensor = []
        for _ in range(numActions):
            init = tf.initializers.random_normal()
            rearrangeLayer = tf.layers.Dense(stateVectorSize,
                                             trainable=trainable,
                                             kernel_initializer=init,
                                             bias_initializer=init)
            tensor = rearrangeLayer.apply(self.state_flat)
            self.rearrangeLayers.append(rearrangeLayer)
            self.arrangementTensor.append(tensor)

        layerSizes = [stateVectorSize,
                      stateVectorSize,
                      stateVectorSize]
        self.commonLayers = DenseLayers(layerSizes,
                                        trainable=trainable,
                                        keep_prob=self.keep_prob)
        self.commonTensors = [self.commonLayers.apply(arTensor)
                              for arTensor in self.arrangementTensor]

        self.commonFinal = self.finalLayer("finalLayer", 1,
                                           trainable=trainable)

        self.outputList = [self.commonFinal.apply(cTensor)
                           for cTensor in self.commonTensors]
        self.actions = tf.layers.flatten(tf.stack(self.outputList, axis=1))

    def get_all_variables(self):
        variables = []
        for denseLayer in self.rearrangeLayers:
            variables.append(denseLayer.kernel)
            variables.append(denseLayer.bias)

        variables += self.commonLayers.allVariables()

        variables.append(self.commonFinal.kernel)
        variables.append(self.commonFinal.bias)

        return variables

    def finalLayer(self, name, num_output, trainable=True):
        init = tf.initializers.random_normal()
        return tf.layers.Dense(units=num_output,
                               kernel_initializer=init,
                               bias_initializer=init,
                               trainable=trainable,
                               name=name)


def mkConstructor(stateDims, numActions):
    def construct(trainable):
        return Rearrange(stateDims=stateDims,
                         numActions=numActions,
                         trainable=trainable)

    return construct
