import tensorflow as tf


class DenseLayers:
    """Defines layers to be applied in a neural net function.
    Arguments
      trainable: if the variables can change from trianing
      keep_prob: tensor to determine the keep probability of the
      dropout layers"""
    def __init__(self,
                 layerSizes,
                 trainable=True,
                 keep_prob=None):
        self.layerSizes = layerSizes
        self.trainable = trainable
        self.keep_prob = keep_prob
        self.build()

    def build(self):
        self.layers = []
        for (layerNumber, layerSize) in enumerate(self.layerSizes):
            layerName = "Layer{}".format(layerNumber)
            layer = self.makeLayer(layerName, layerSize)
            # Tensor output is None until application
            self.layers.append((layer, None))

    def apply(self, input):
        lastTensor = input
        updatedLayers = []
        for (layer, _) in self.layers:
            lastTensor = self.applyLayer(layer, lastTensor)
        return lastTensor

        self.layers = updatedLayers
        return lastTensor

    def makeLayer(self, name, num_output):
        init = tf.initializers.random_normal()
        return tf.layers.Dense(units=num_output,
                               activation=tf.nn.softplus,
                               kernel_initializer=init,
                               bias_initializer=init,
                               trainable=self.trainable,
                               name=name)

    def applyLayer(self, layer, input):
        output = layer.apply(input)
        if self.keep_prob is not None:
            output = tf.nn.dropout(output, self.keep_prob)
        return output

    def allVariables(self):
        vars = []
        for (layer, _) in self.layers:
            vars.append(layer.kernel)
            vars.append(layer.bias)
        return vars
