This is a tensorflow implementation of the VGG16 network by the Visual Geometery Group at Oxford. We have converted weights from their [creative commons caffemodel](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for use in tensorflow.

### Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### Installation

```
pip install tfnets
```

### Usage

#### Simple instantiation and feed forward run:

``` python
import tensorflow as tf
from tfnets import vgg16

# create vgg16 graph operations
graph = tf.Graph()
with graph.as_default():
    input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.build(input_tensor=input_tensor)
    init = tf.initialize_all_variables()

# start a session and run graph on blank image
sess = tf.Session()
sess.run(init)
inputs = {input_tensor: np.zeros([1,224,224,3])}
outputs = [net.pred_softmax]
print(sess.run(outputs, feed_dict=inputs))
```

The `nets` variable in this case is a [attribute dict](http://stackoverflow.com/a/14620633) that contains all layers of the VGG16 network. You can see a list of layers by calling `nets.keys()`. You can access any layer by using dot notation e.g: `net.conv1_1`, `net.fc8`,  and so on ...

#### Loading a VGG16 network pretrained on imagenet

First, download the pretrained weights for vgg16 into your working directory:
* [imagenet pretrained weights for vgg16](https://drive.google.com/open?id=0B7Q0GPJoPX8aSDhVUW9xZHpPcVk)

Then use the following code to load the weights and run a forward pass.

``` python
import tensorflow as tf
from tfnets import vgg16

# create vgg16 graph operations
graph = tf.Graph()
with graph.as_default():
    inp = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.build(input_tensor=inp)


# start a session and populate vgg variables with pretrained weights
sess = tf.Session(graph=graph)
vgg16.restore(sess, fpath="imagenet_trained_vgg16.npy")

# compute softmax on blank image
inputs = {input_tensor: np.zeros([1,224,224,3])}
outputs = [net.pred_softmax]
print(sess.run(outputs, feed_dict=inputs))

```
