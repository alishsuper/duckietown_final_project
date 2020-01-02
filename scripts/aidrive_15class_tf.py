#!/usr/bin/python

## Build in
import os
from argparse import ArgumentParser

## Installed
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def init_argparser(parents=[]):
    '''
    Initialize an ArgumentParser for this script.
    Args:
        parents: A list of ArgumentParsers of other scripts, if any.
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description='Trains a ai-driving model based on 15-class steering angles.',
        parents=parents
        )

    parser.add_argument(
        '--train_files', '-X',
        help='One or more index-file of training samples',
        nargs='+'
        )
    
    parser.add_argument(
        '--test_files', '-Y',
        help='One or more index-file of validation samples',
        nargs='*'
        )

    parser.add_argument(
        '--x_header', '-x',
        help='CSV header(s) for input data',
        default='image'
        )
    
    parser.add_argument(
        '--t_header', '-t',
        help='CSV header(s) for label data',
        default='omega'
        )

    parser.add_argument(
        '--outdir', '-o',
        help='The output folder where the model and checkpoints get stored at',
        default=None
        )
    
    parser.add_argument(
        '--name', '-N',
        help='Name of the model',
        default='aidrive15'
        )

    parser.add_argument(
        '--epochs', '-e',
        help='How many epochs to repeat the dataset',
        type=int,
        default=1
        )
    
    parser.add_argument(
        '--batch', '-b',
        help='How many samples per batch',
        type=int,
        default=32
        )
    
    parser.add_argument(
        '--classes', '-K',
        help='The number of classes',
        type=int,
        default=15
        )
    
    parser.add_argument(
        '--lr', '-l',
        help='The learning rate',
        type=float,
        default=1e-4
        )
    
    parser.add_argument(
        '--dropout', '-d',
        help='The dropout ratio',
        type=float,
        default=0.7
        )
    
    parser.add_argument(
        '--shape', '-s',
        help='The input shape (samples, height, width, channels)',
        type=int,
        nargs=4,
        default=(1, 101, 101, 3)
        )
    
    parser.add_argument(
        '--checkpoints', '-c',
        help='Create a checkpoint for each step (-c)',
        default=0
        )
    
    parser.add_argument(
        '--model', '-m',
        help='Path to a pre-trained model',
        default=None
        )
    
    parser.add_argument(
        '--delimiter', '-D',
        help='Delimiter of the csv file',
        default=None
        )
    
    parser.add_argument(
        '--export', '-E',
        help='Export the graph',
        type=bool,
        const=True,
        nargs='?',
        default=False
        )

    return parser


class MultiIndexDatagenerator():
    def __init__(self, index_files, x_header, t_header, classes=15, shape=None, fit=True, delimiter=None):
        self.index_files = index_files
        self.shape = shape
        self.fit = fit
        self.step = 0
        
        self.load_index_files(index_files, x_header, t_header, classes, delimiter)
        pass
    
    def load_index_files(self, index_files, x_header, t_header, classes=15, delimiter=None):
        df = pd.concat((pd.read_csv(index_file, delimiter=delimiter) for index_file in index_files))
        t = df[t_header].to_numpy()
        t = (t / (t.max() - t.min()) * classes).astype(int) + np.floor(classes * 0.5)
        self.t = tf.keras.utils.to_categorical(t, classes)
        self.x = df[x_header]
        pass
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        self.step += 1
        x = np.array(cv2.imread(self.x[index]), dtype=np.float32) / 255
        t = np.array(self.t[index])
        
        if self.shape:
            x = cv2.resize(x, self.shape[1:3], interpolation=cv2.INTER_CUBIC)
            pass
        
        if self.fit:
            epoch = int(self.step / len(self)) + 1
            return x, t, index, self.step, epoch
        else:
            return x
    
    def __call__(self):
        for i in range(len(self)):
            yield self[i]


class Classifier():
    def __init__(self, x=None, classes=15, shape=(None ,101, 101, 3), dropout=0.7):
        self.classes = classes
        self.target = None
    
        if x:
            self.input = x
        else:
            self.input = tf.placeholder(tf.float32, shape=shape, name='input')
    
        with tf.name_scope('classifier'):
            self.conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08, dtype=tf.float32))
            self.conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08, dtype=tf.float32))
            self.conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08, dtype=tf.float32))
            self.conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08, dtype=tf.float32))

            # 1, 2
            self.conv1 = tf.nn.conv2d(self.input, self.conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_relu = tf.nn.relu(self.conv1)
            self.conv1_pool = tf.nn.max_pool(self.conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.conv1_bn = tf.layers.batch_normalization(self.conv1_pool, fused=False)

            # 3, 4
            self.conv2 = tf.nn.conv2d(self.conv1_bn, self.conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_relu = tf.nn.relu(self.conv2)
            self.conv2_pool = tf.nn.max_pool(self.conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.conv2_bn = tf.layers.batch_normalization(self.conv2_pool, fused=False)

            # 5, 6
            self.conv3 = tf.nn.conv2d(self.conv2_bn, self.conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_relu = tf.nn.relu(self.conv3)
            self.conv3_pool = tf.nn.max_pool(self.conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.conv3_bn = tf.layers.batch_normalization(self.conv3_pool, fused=False)

            # 7, 8
            self.conv4 = tf.nn.conv2d(self.conv3_bn, self.conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_relu = tf.nn.relu(self.conv4)
            self.conv4_pool = tf.nn.max_pool(self.conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.conv4_bn = tf.layers.batch_normalization(self.conv4_pool, fused=False)

            # 9
            self.flat = tf.contrib.layers.flatten(self.conv4_bn)

            # 10
            self.full1 = tf.contrib.layers.fully_connected(inputs=self.flat, num_outputs=1024, activation_fn=tf.nn.relu)
            if dropout:
                self.full1 = tf.nn.dropout(self.full1, dropout)
            self.full1_bn = tf.layers.batch_normalization(self.full1, fused=False)

            '''
            # 11
            self.full2 = tf.contrib.layers.fully_connected(inputs=self.full1_bn, num_outputs=256, activation_fn=tf.nn.relu)
            self.full2_drop = tf.nn.dropout(self.full2, dropout)
            self.full2_bn = tf.layers.batch_normalization(self.full2_drop, fused=False)

            # 12
            self.full3 = tf.contrib.layers.fully_connected(inputs=self.full2, num_outputs=512, activation_fn=tf.nn.relu)
            self.full3_drop = tf.nn.dropout(self.full3, dropout)
            self.full3_bn = tf.layers.batch_normalization(self.full3_drop, fused=False)

            # 13
            self.full4 = tf.contrib.layers.fully_connected(inputs=self.full3, num_outputs=1024, activation_fn=tf.nn.relu)
            self.full4_drop = tf.nn.dropout(self.full4, dropout)
            self.full4_bn = tf.layers.batch_normalization(self.full4_drop, fused=False)
            '''

        # 14
        self.logits = tf.contrib.layers.fully_connected(inputs=self.full1_bn, num_outputs=self.classes, activation_fn=None)
        self.output = tf.nn.softmax(self.logits, name='output')
        pass
        
    def setup_train_pip(self, t=None, lr=1e-4):
        if t:
            self.target = t
        else:
            self.target = tf.placeholder(tf.float32, shape=self.logits.shape, name='target')
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        pass
    
    def setup_test_pip(self, t=None):
        if t:
            self.target = t
        elif self.target is None:
            self.target = tf.placeholder(tf.float32, shape=self.logits.shape, name='target')
            
        self.cost = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.cost, tf.float32), name='accuracy')
        pass
        
def test(sess, model, test_sample):
    cost_log = []
    acc_log = []
    
    try:
        while True:
            x, t, index, step, epoch = sess.run(test_sample)
            cost, acc = sess.run((model.cost, model.accuracy),
                feed_dict={
                    model.input: x,
                    model.target: t
                })
            cost_log.append(cost.mean())
            acc_log.append(acc)
            print("(Validation) Epoch: {}, Step: {}, Cost {}, Batch Acc: {}".format(epoch.max(), step.max(), loss.mean(), acc))
  
    except tf.errors.OutOfRangeError: #data generator expiered
        print("Done!!!")
        pass
    except KeyboardInterrupt:
        print("Validation aborted!!!")
        pass
    
    final_cost = np.array(cost_log).mean()
    final_acc = np.array(acc_log).mean()
    return final_cost, final_acc


def train(model, train_set, test_set=None, outd=None, name='model', chpt=0):
    train_iter = train_set.make_one_shot_iterator()
    train_sample = train_iter.get_next()
    
    if test_set:
        test_iter = test_set.make_initializable_iterator()
        test_sample = test_iter.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_epoch = 0
        curr_epoch = 1
        
        try:
            while True:
                x, t, index, step, epoch = sess.run(train_sample)
                loss, acc, _ = sess.run((model.loss, model.accuracy, model.optimizer),
                    feed_dict={
                        model.input: x,
                        model.target: t
                    })
                print("(Training) Epoch: {}, Step: {}, Loss {}, Batch Acc: {}".format(epoch.max(), step.max(), loss.mean(), acc))
                
                if test_set:
                    if curr_epoch != epoch.max():
                        sess.run(test_iter.initializer)
                        fin_cost, fin_acc = test(sess, model, test_sample)
                        print("(Validation Result) Final Cost: {}, Final Acc: {}".format(fin_cost, fin_acc))
                        if outd and (best_epoch == 0 or best_epoch > fin_acc):
                            save_path = saver.save(sess, os.path.join(outd, '{}_best'.format(name)))
                            print("Saved best session to:", save_path)
                
                elif outd:
                    if chpt and np.any(np.mod(step, chpt) == 0):
                        save_path = saver.save(sess, os.path.join(outd, '{}_{}'.format(name, step.max())))
                        print("Saved session to:", save_path)
                        
                    
        except tf.errors.OutOfRangeError: #data generator expiered
            print("Done!!!")
            pass
        except KeyboardInterrupt:
            print("Training aborted!!!")
            pass
        
        if outd:
            save_path = saver.save(sess, os.path.join(outd, name))
            print("Saved session to:", save_path)


def main(args):
    if args.train_files:
        print("Prepare training data...")
        train_index = MultiIndexDatagenerator(
                args.train_files, 
                args.x_header, 
                args.t_header,
                args.classes,
                args.shape,
                fit=True,
                delimiter=args.delimiter
                )
        train_samples = len(train_index)
        train_set = tf.data.Dataset.from_generator(train_index, output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32))
        train_set = train_set.shuffle(args.batch).repeat(args.epochs).batch(args.batch)
    else:
        train_set = None
    
    if args.test_files:
        print("Prepare test data...")
        test_index = MultiIndexDatagenerator(
            args.test_files, 
            args.x_header, 
            args.t_header,
            args.classes,
            args.shape,
            fit=True,
            delimiter=args.delimiter
            )
        test_samples = len(test_index)
        test_set = tf.data.Dataset.from_generator(test_index, output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32))
    else:
        test_set = None

    if args.model: #load model
        print("Load model...")
        pass
    else: # init model
        print("Initialize new model...")
        shape = (None, args.shape[1], args.shape[2], args.shape[3])
        model = Classifier(None, args.classes, shape, args.dropout)
        if train_set:
            model.setup_train_pip(None, args.lr)
            model.setup_test_pip()
        elif test_set:
            model.setup_test_pip()
        pass

    if train_set:
        print("Start training...")
        train(
            model,
            train_set,
            test_set,
            args.outdir,
            args.name,
            len(train_index) if args.checkpoints == 'epoch' else int(args.checkpoints)
            )
        
    elif test_set: #run test only
        print("Start testing...")
        test_iter = test_set.make_initializable_iterator()
        test_sample = test_iter.get_next()
        
        with tf.Session() as sess:
            sess.run(test_iter.initializer)
            test(
                sess,
                model,
                test_sample
                )
    
    if args.export and args.outdir:
        tf.reset_default_graph()
        shape = (1, args.shape[1], args.shape[2], args.shape[3])
        model = Classifier(None, args.classes, shape, 0)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, os.path.join(args.outdir, args.name))
            save_path = saver.save(sess, os.path.join(args.outdir, "{}_freezed".format(args.name)))
        print("Saved freezed graph to:", save_path)
    return 0

if __name__ == '__main__':
    parser = init_argparser()
    args, _ = parser.parse_known_args()
    e = main(args)
    exit(e)