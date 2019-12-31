#!/usr/bin/python

## Build in
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
        '--t_headers', '-t',
        help='CSV header(s) for label data',
        default='omega',
        nargs='+'
        )

    parser.add_argument(
        '--outdir', '-o',
        help='The output folder where the model and checkpoints get stored at',
        default='./models'
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
        help='The dropout ratio',
        type=int,
        nargs=3,
        default=(101, 101, 3)
        )
    
    parser.add_argument(
        '--checkpoints', '-c',
        help='Create a checkpoint for each step (-c)',
        type=int,
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

    return parser


class MultiIndexDatagenerator():
    def __init__(self, index_files, x_header, t_headers, shape=None, fit=True, delimiter=None):
        self.index_files = index_files
        self.x_header = x_header
        self.t_headers = t_headers
        self.shape = shape
        self.fit = fit
        self.delimiter = delimiter
        
        self.load_index_files(index_files)
        pass
    
    def load_index_files(self, index_files):
        self.df = pd.concat((pd.read_csv(index_file, delimiter=self.delimiter) for index_file in index_files))
        pass
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        img_path = self.df.at[index, self.x_header]
        x = np.array(cv2.imread(img_path), dtype=np.float32) / 255
        t = np.array(self.df.at[index, self.t_headers[0]])
        
        if self.shape:
            x = cv2.resize(x, self.shape[:2], interpolation=cv2.INTER_CUBIC)
            pass
        
        if self.fit:
            return x[None, :], t
        else:
            return x[None, :]
    
    def __call__(self):
        for i in range(len(self)):
            yield self[i]


class Classifier():
    def __init__(self, x=None, classes=15, shape=(101, 101, 3), dropout=0.7):
        if x:
            self.input = x
        else:
            self.input = tf.placeholder(tf.float32, shape=(None, shape[0], shape[1], shape[2]), name='input')
    
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
        self.full1 = tf.contrib.layers.fully_connected(inputs=self.flat, num_outputs=128, activation_fn=tf.nn.relu)
        self.full1_drop = tf.nn.dropout(self.full1, dropout)
        self.full1_bn = tf.layers.batch_normalization(self.full1_drop, fused=False)

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

        # 14
        self.logits = tf.contrib.layers.fully_connected(inputs=self.full4_bn, num_outputs=classes, activation_fn=None)
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
    
    def setup_test_pip(self):
        self.cost = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.cost, tf.float32), name='accuracy')
        pass
        
def test():
    pass


def train(model, train_set, test_set=None, outd='./', name='model', chpt=0, samples=0):
    train_iter = train_set.make_one_shot_iterator()
    train_x, train_t = train_iter.get_next()
    train_t = tf.cast(train_t, tf.int32)
    
    if test_set:
        test_iter = test_set.make_one_shot_iterator()
        test_x, test_t = test_iter.get_next()
        test_t = tf.cast(t, tf.int32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        try:
            step = 0
            while True:
                x, t = sess.run((train_x, train_t))
                loss = sess.run(model.optimizer,
                        feed_dict={
                            model.input: x,
                            model.target: t
                        })
                print("Step:", step, "Loss:", loss.shape)
                step += 1
        except tf.errors.OutOfRangeError: #data generator expiered
            print("Done!!!")
            pass
        except KeyboardInterrupt:
            print("Training aborted!!!")
            pass


def main(args):
    if args.train_files:
        print("Prepare training data...")
        train_index = MultiIndexDatagenerator(
                args.train_files, 
                args.x_header, 
                args.t_headers, 
                args.shape,
                fit=True,
                delimiter=args.delimiter
                )
        train_samples = len(train_index)
        train_set = tf.data.Dataset.from_generator(train_index, output_types=(tf.float32, tf.int32))
        train_set.shuffle(args.batch).batch(args.batch).repeat(args.epochs)
        print(train_index.df)
    else:
        train_set = None
    
    if args.test_files:
        print("Prepare test data...")
        test_set = tf.data.Dataset.from_generator(
            MultiIndexDatagenerator(
                args.test_files, 
                args.x_header, 
                args.t_headers, 
                args.shape, 
                fit=True,
                delimiter=args.delimiter
                ).gen_data()
            )
    else:
        test_set = None

    if args.model: #load model
        print("Load model...")
        pass
    else: # init model
        print("Initialize new model...")
        model = Classifier(None, args.classes, args.shape, args.dropout)
        if train_set:
            model.setup_train_pip(None, args.lr)
        if test_set:
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
            args.checkpoints,
            train_samples
            )
    elif test_set: #run test only
        print("Start testing...")
        test()
    return 0

if __name__ == '__main__':
    parser = init_argparser()
    args, _ = parser.parse_known_args()
    e = main(args)
    exit(e)