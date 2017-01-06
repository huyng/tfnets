
This is an implementation of VGG16

#### Usage

##### Simple instantiation and feed forward run:

``` python
import tensorflow as tf
import tfnets.vgg16 as vgg16

# build vgg16 graph
graph = tf.Graph()
with graph.as_default():
    input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.build(input_tensor=input_tensor)

# run graph on blank image
sess = tf.Session()
inputs = {input_tensor: np.zeros([1,224,224,3])}
outputs = [net.pred_softmax]
print(sess.run(outputs, feed_dict=inputs))
```

The `nets` variable in this case is a dictionary that contains all layers of the VGG16 network. You can see a list of all layers by calling nets.keys().

##### Loading a VGG16 network pretrained on imagenet

``` python
import tensorflow as tf
import tfnets.vgg16 as vgg16


graph = tf.Graph()
with graph.as_default():
    inp = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.build(input_tensor=inp)


# we initialize a network from weights pretrained on imagenet
sess = tf.Session(graph=graph)
vgg16.restore(sess, fpath="imagenet_trained_vgg16.npy")

# compute softmax on blank image
inputs = {input_tensor: np.zeros([1,224,224,3])}
outputs = [net.pred_softmax]
print(sess.run(outputs, feed_dict=inputs))

```

##### Full image classification example

```
# compute softmax on real image
import io
import numpy as np
from urllib2 import urlopen
from PIL import Image

def center_crop(pimg):
    w, h = pimg.size
    s = min([w,h])
    x0 = (w - s)/2
    y0 = (h - s)/2
    x1 = (w + s)/2
    y1 = (h + s)/2
    return pimg.crop([x0, y0, x1, y1])

image = urlopen("https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg")
image = image.read()
pimg = Image.open(io.BytesIO(image)).convert(mode="RGB")
pimg = center_crop(pimg)
pimg = pimg.resize([224,224])
nimg = np.array(pimg, dtype=np.float32)
nimg = np.expand_dims(nimg, 0)

feed_dict = {}
feed_dict[inp] =  nimg
predictions = sess.run(net.pred_softmax, feed_dict=feed_dict)
```
