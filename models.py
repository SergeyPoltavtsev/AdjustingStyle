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
             .conv(3, 3, 128, 256, name='conv3_1')
             .conv(3, 3, 256, 256, name='conv3_2')
             .conv(3, 3, 256, 256, name='conv3_3')
             .pool(name = 'pool3')
             .conv(3, 3, 256, 512, name='conv4_1')
             .conv(3, 3, 512, 512, name='conv4_2')
             .conv(3, 3, 512, 512, name='conv4_3')
             .pool(name = 'pool4')
             .conv(3, 3, 512, 512, name='conv5_1')
             .conv(3, 3, 512, 512, name='conv5_2')
             .conv(3, 3, 512, 512, name='conv5_3')
             .pool(name = 'pool5')
             .fc(name='fc6', withRelu=True)
             .fc(name='fc7', withRelu=True)
             .fc(name='fc8', withRelu=False)
        )

    def y(self):
        return self.vardict['fc7']

def getModel(image, params_path):
    return VGG16(image, params_path)
