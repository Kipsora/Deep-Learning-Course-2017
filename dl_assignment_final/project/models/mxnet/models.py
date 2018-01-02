import mxnet

from project.hparam import get_hparam
from project.models.mxnet.base import MXModel


class DeepNN(MXModel):
    def __init__(self, name, ishape, osize, hparam=None, path='./save'):
        super(DeepNN, self).__init__(name, hparam, path)

    def _forward(self, F, input):
        hidden = input
        for layer in self._layers:
            hidden = layer(hidden)
        return hidden

    def _declare(self):
        self._layers = []
        for layer in self._hparam.layers:
            if layer.batchnorm:
                self._layers.append(mxnet.gluon.nn.BatchNorm(**layer.batchnorm))
            self._layers.append(mxnet.gluon.nn.Dense(
                units=layer.units,
                activation=layer.activation
            ))
        return self._layers

    @staticmethod
    def default_hparam():
        result = get_hparam(
            layers=[
                get_hparam(
                    units=20,
                    activation='relu',
                    batchnorm=get_hparam()
                ),
                get_hparam(
                    units=20,
                    activation='relu',
                    batchnorm=get_hparam()
                )
            ],
            reg=get_hparam(
                type='L2',
                value=0.001
            ),
            loss='mean_squared_error',
            optimizer=get_hparam(
                type='RMSProp',
                params=get_hparam(
                    learning_rate=0.001
                )
            )
        )
        result.update(DeepNN.__bases__[0].default_hparam())
        return result
