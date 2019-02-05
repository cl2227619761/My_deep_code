from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


train_filenames = [os.path.join("./train_tfrecords", i) for i in os.listdir("./train_tfrecords")]
val_filenames = [os.path.join("./test_tfrecords", i) for i in os.listdir("./test_tfrecords")]


def _parse_function(record):
    keys_to_features = {"img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                        "label": tf.FixedLenFeature((), tf.int64,
                                                    default_value=tf.zeros([], tf.int64))}
    parsed = tf.parse_single_example(record, keys_to_features)
    images = tf.decode_raw(parsed["img_raw"], tf.uint8)
    images = tf.reshape(images, [784])
    images = tf.cast(images, tf.float32)
    images = tf.divide(images, 255.)
    labels = tf.one_hot(parsed["label"], 10)
    labels = tf.cast(labels, tf.int64)
    return images, labels


train_ds = tf.data.TFRecordDataset(train_filenames)
train_ds = train_ds.map(_parse_function)
train_ds = train_ds.shuffle(10000).repeat(5).batch(100)

val_ds = tf.data.TFRecordDataset(val_filenames)
val_ds = val_ds.map(_parse_function)
val_ds = val_ds.repeat(5).batch(100)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,
                                               train_ds.output_types, train_ds.output_shapes)
next_element = iterator.get_next()
training_iterator = train_ds.make_one_shot_iterator()
validation_iterator = val_ds.make_initializable_iterator()


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weight"):
            weights = weight_variable(shape=[input_dim, output_dim])
        with tf.name_scope("biases"):
            biases = bias_variable(shape=[output_dim])
        with tf.name_scope("Wx_plus_b"):
            preactive = tf.add(tf.matmul(input_tensor, weights), biases)
            tf.summary.histogram("pre_activation", preactive)
        activation = act(preactive, name="activation")
        tf.summary.histogram("activation", activation)
        return activation


hidden1 = nn_layer(next_element[0], 784, 512, "layer1")

with tf.name_scope("dropout"):
    dropped = tf.nn.dropout(hidden1, 0.8)


y = nn_layer(dropped, 512, 10, "layer2", act=tf.identity)


with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(next_element[1], logits=y))
tf.summary.scalar("cross_entropy", cross_entropy)


with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(next_element[1], 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)


merged = tf.summary.merge_all()
init = tf.global_variables_initializer()


with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init)
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "caolei-Inspiron-7420:7000")
    train_writer = tf.summary.FileWriter("./log/train/", sess.graph)
    test_writer = tf.summary.FileWriter("./log/test")
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    while not sess.should_stop():
        for epoch in range(5):
             for i in range(600):
                summary, _, train_acc = sess.run([merged, train_step, accuracy], feed_dict={handle: training_handle})
                if (i+1) % 600 == 0:
                    train_writer.add_summary(summary, epoch)
                    sess.run(validation_iterator.initializer)
                    for j in range(100):
                        val_summary, val_acc = sess.run([merged, accuracy], feed_dict={handle: validation_handle})
                        if (j+1) % 100 == 0:
                            test_writer.add_summary(val_summary, epoch)
             print("Epoch %s: train acc %s, val acc %s" % (epoch, train_acc, val_acc))
    train_writer.close()
    test_writer.close()
