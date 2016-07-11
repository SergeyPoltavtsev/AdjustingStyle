from network import Network

class VGG16(Network):
    alpha = [0, 0, 0, 1, 1]
    beta  = [1, 1, 1, 1, 1]
    def setup(self):
        (self.conv(3, 3,   3,  64, name='conv1_1')
             .conv(3, 3,  64,  64, name='conv1_2')
             .pool(name = 'pool1')
             .conv(3, 3,  64, 128, name='conv2_1')
             .conv(3, 3, 128, 128, name='conv2_2')
             .pool(name = 'pool2')
        )

    def y(self):
        return self.vardict['pool2']

def getModel(image, params_path):
    return VGG16(image, params_path)
