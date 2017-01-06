
This is an implementation of VGG16
``` python
import tensorflow as tf
import tfnets.vgg16 as vgg16

# BUILD GRAPH
# ===========

# you can generate the vgg16 graph operations
graph = tf.Graph()
with graph.as_default():
    inp = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = vgg16.build(input_tensor=inp)

# access the fc8 layer tensor
print(net.fc8)

# access the softmax outputs tensor
print(net.pred_softmax)

# show all available layers
net.keys()

# LOAD WEIGHTS
# ============

# we can initialize the vgg16 network using
# weights pre-trained on imagenet
sess = tf.Session(graph=graph)
vgg16.restore(sess, fpath="imagenet_trained_vgg16.npy")


# RUN FORWARD PASS
# ================

# compute softmax on blank image
feed_dict = {inp: np.zeros([0, 224, 224, 3])}
predictions = sess.run(net.pred_softmax, feed_dict=feed_dict)

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
