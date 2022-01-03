import tensorflow as tf

from tensorflow.keras import Model, layers

class mCNN(Model):
  def __init__(self,
               window_sizes=[8,16,24,32,40,48],
               max_length=5000,
               num_feature=20,
               num_filters=256,
               num_hidden=1024,
               num_class=2):
    super(mCNN, self).__init__()
    # Add input layer
    input_shape=(1,max_length,num_feature)
    self.input_layer = tf.keras.Input(input_shape)
    self.window_sizes = window_sizes
    self.conv2d = []
    self.maxpool = []
    self.flatten = []
    for window_size in self.window_sizes:
      self.conv2d.append(layers.Conv2D(
        filters=num_filters,
        kernel_size=(1,window_size),
        activation=tf.nn.relu,
        padding='valid',
        bias_initializer=tf.constant_initializer(0.1),
        kernel_initializer=tf.keras.initializers.GlorotUniform()
      ))
      self.maxpool.append(layers.MaxPooling2D(
          pool_size=(1,MAX_LENGTH - window_size + 1),
          strides=(1,MAX_LENGTH),
          padding='valid'))
      self.flatten.append(layers.Flatten())
    self.dropout = layers.Dropout(rate=0.7)
    self.fc1 = layers.Dense(
      num_hidden,
      activation=tf.nn.relu,
      bias_initializer=tf.constant_initializer(0.1),
      kernel_initializer=tf.keras.initializers.GlorotUniform()
    )
    self.fc2 = layers.Dense(num_class,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    # Get output layer with `call` method
    self.out = self.call(self.input_layer)


  def call(self, x, training=False):
    _x = []
    for i in range(len(self.window_sizes)):
      x_conv = self.conv2d[i](x)
      x_maxp = self.maxpool[i](x_conv)
      x_flat = self.flatten[i](x_maxp)
      _x.append(x_flat)

    x = tf.concat(_x,1)
    x = self.dropout(x,training=training)
    x = self.fc1(x)
    x = self.fc2(x)
    return x