from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow.keras as tk
#https://github.com/keras-team/keras/blob/master/keras/layers/preprocessing/image_preprocessing.py#L316-L361
class Rescaling(tk.layers.Layer):
  """Multiply inputs by `scale` and adds `offset`.
  For instance:
  1. To rescale an input in the `[0, 255]` range
  to be in the `[0, 1]` range, you would pass `scale=1./255`.
  2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
  you would pass `scale=1./127.5, offset=-1`.
  The rescaling is applied both during training and inference.
  Input shape:
    Arbitrary.
  Output shape:
    Same as input.
  Args:
    scale: Float, the scale to apply to the inputs.
    offset: Float, the offset to apply to the inputs.
  """

  def __init__(self, scale, offset=0., **kwargs):
    self.scale = scale
    self.offset = offset
    super(Rescaling, self).__init__(**kwargs)

  def call(self, inputs):
    return inputs * self.scale + self.offset

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'scale': self.scale,
        'offset': self.offset,
    }
    base_config = super(Rescaling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))