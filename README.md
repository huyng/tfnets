TFNets is a library of common computer vision models implemented in tensorflow.

### installation

```
pip install tfnets
```

### models & usage instructions

* [vgg16](https://github.com/huyng/tfnets/blob/master/tfnets/vgg16)

---

### Rationale

We are not the first to aggregate these models into a "model zoo". Many other implementations of these models exist around the web.

But unlike other implementations, the major goal for this project is to promote and enable the concept of "reusable" tensorflow models by following a few guiding principles.

What makes a tensorflow model "reusable" you ask? We think it's the following:

#### Tenets of a "reusable" tensorflow model

* **Models should be pip installable.** Rationale: You shouldn't have to rework your entire project structure to try out a new model. Trying a new model should ideally just be two steps:  1) `pip install your.model` 2) `import your.model`

* **Models should separate graph definition from graph execution **. Rationale: Don't mix code that requires a `tf.Session` with code that is just defining your graph operations. Users of your model should be able to instantiate your graph and run it within their *own* `tf.Session`. Structuring it this way,  gives them an opportunity to augment/customize your graph before running it.

* **Models should expose references to all significant operation outputs** Rationale: Your users may be using your models for unanticipated tasks. Give them easy access to intermediate outputs and the underlying variables.
