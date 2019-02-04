from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os


def train():
    def _parse_function(record):
        """tfrecord文件解析函数"""
        keys_to_features = {"img_raw": tf.FixedLenFeature((), tf.string, default_value=""), 
        "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], tf.int64))}
        parsed = tf.parse_single_example(record, keys_to_features)
        images = tf.decode_raw(parsed["img_raw"], tf.uint8)
        images = tf.reshape(images, [784])
        images = tf.cast(images, tf.float32)
        images = tf.divide(images, 255.)
        labels = tf.one_hot(parsed["label"], 10)
        labels = tf.cast(labels, tf.int64)
        return images, labels


    def prepare_dataset(filepath, batch_size=100):
        """数据集生成函数"""
        filenames = [os.path.join(filepath, i) for i in os.listdir(filepath)]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset


    train_ds = prepare_dataset("./train_tfrecords")
    val_ds = prepare_dataset("./test_tfrecords")
    iteration = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    next_element = iteration.get_next()

    train_init_op = iteration.make_initializer(train_ds)
    val_init_op = iteration.make_initializer(val_ds)


    def weight_variable(shape):
        """定义权重函数"""
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        """定义偏置函数"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def variable_summaries(var):
        """定义变量的summary"""
        with tf.name_scope("summaries"):
            with tf.name_scope("mean"):
                mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

    
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                weights = weight_variable(shape=[input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope("biases"):
                biases = bias_variable(shape=[output_dim])
                variable_summaries(biases)
            with tf.name_scope("Wx_plus_b"):
                preactivate = tf.add(tf.matmul(input_tensor, weights), biases)
                tf.summary.histogram("pre_activate", preactivate)
            activations = act(preactivate, name="activations")
            tf.summary.histogram("activations", activations)
            return activations
    

    hidden1 = nn_layer(next_element[0], 784, 512, "layer1")


    with tf.name_scope("dropout"):
        dropped = tf.nn.dropout(hidden1, 0.2)

    
    y = nn_layer(dropped, 512, 10, "layer2", act=tf.identity)

    with tf.name_scope("cross_entropy"):
        with tf.name_scope("total"):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=next_element[1], logits=y)
    tf.summary.scalar("cross_entropy", cross_entropy)
    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(next_element[1], 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)
        sess = tf_debug.TensorBoardDebugWrapperSession(sess, "caolei-Inspiron-7420:7000")
        train_writer = tf.summary.FileWriter("./log/train", sess.graph)
        test_writer = tf.summary.FileWriter("./log/test")
        sess.run(train_init_op)
        for i in range(300):
            # if i % 10 == 0:
            #     sess.run(val_init_op)
            #     summary, acc = sess.run([merged, accuracy])
            #     test_writer.add_summary(summary, i)
            #     print("Accuracy at step %s: %s" % (i, acc))
            # else:
            #     sess.run(train_init_op)
            if i % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step%03d" % i)
                train_writer.add_summary(summary, i)
                print("Adding run metadata for", i)
            else:
                summary, _ = sess.run([merged, train_step])
                train_writer.add_summary(summary, i)
        sess.run(val_init_op)
        for i in range(300):
            sess.run(val_init_op)
            summary, acc = sess.run([merged, accuracy])
            test_writer.add_summary(summary, i)
            print("Accuracy at step %s: %s" % (i, acc))
        train_writer.close()
        test_writer.close()


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run(main=main)
