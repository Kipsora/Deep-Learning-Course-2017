from project.models.mxnet.base import MXModel


class DeepNN(MXModel):
    def __init__(self, name, ishape, osize, hparam=None, path='./save'):
        super(DeepNN, self).__init__(name, ishape, osize, hparam, path)

    def predict(self, X):
        pass

    def loss(self, X, y):
        pass

    def train(self, X, y):
        pass

    def _build(self, net, ishape, osize):
        pass
