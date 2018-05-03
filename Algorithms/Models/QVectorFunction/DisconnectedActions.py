import tensorflow as tf

from Algorithms.Models.QVectorFunction.BoardRepresentations import OneHotBoard
from util.tensorflow.DenseLayers import DenseLayers


class DisconnectedActions:

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

        self.boardRep = OneHotBoard(stateDims)
        self.state = self.boardRep.input_state

        [_, stateVectorSize] = self.boardRep.state_representation.get_shape()
        layerSizes = [stateVectorSize for _ in range(3)]

        self.denseNetworks = []
        self.denseTensors = []
        self.actionLayers = []
        self.actionTensors = []
        init = tf.initializers.random_normal()
        for _ in range(numActions):
            denseNet = DenseLayers(layerSizes, trainable, self.keep_prob)
            t = denseNet.apply(self.boardRep.state_representation)
            self.denseNetworks.append(denseNet)
            self.denseTensors.append(t)
            outLayer = tf.layers.Dense(units=1,
                                       kernel_initializer=init,
                                       bias_initializer=init,
                                       trainable=trainable)
            outT = outLayer.apply(t)
            self.actionLayers.append(outLayer)
            self.actionTensors.append(outT)

        self.actions = tf.layers.flatten(tf.stack(self.actionTensors, axis=1))

    def get_all_variables(self):
        allvars = []
        for denseNet in self.denseNetworks:
            allvars += denseNet.allVariables()
        allvars += [l.kernel for l in self.actionLayers]
        allvars += [l.bias for l in self.actionLayers]
        return allvars


def mkConstructor(stateDims, numActions):
    def construct(trainable):
        return DisconnectedActions(stateDims=stateDims,
                                   numActions=numActions,
                                   trainable=trainable)

    return construct
