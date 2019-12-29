## Build in
from argparse import ArgumentParser

## Install
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tf.keras.layers import Conv2D, MaxPooling2D
from tf.keras import regularizers
from tf.keras.utils import Sequence


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
        '--train', '-t',
        help='The one or more index-file of training samples.',
        nargs='+'
        )
    
    parser.add_argument(
        '--val', '-v',
        help='The one or more index-file of validation samples.',
        nargs='+'
        )

    parser.add_argument(
        '--outdir', '-o',
        help='The output folder where the model and checkpoints get stored at.',
        default='./models'
        )
    
    parser.add_argument(
        '--name', '-N',
        help='Name of the model.',
        default='aidrive15'
        )

    parser.add_argument(
        '--epochs', '-e',
        help='Name of the model.',
        type=int
        default=1
        )

    return parser


def main(args):
    

    return 0

# 1. data preprocessing - get training data
x_train = []
x_test = []
# 3570: the size of data_set
shape101 = 101
for i in range(3570):
    # show progress
    if i % 50 == 0:
        print(i, end="\r")
    temp = cv2.imread('ele/' + str(i) + '.jpg')
    data = cv2.resize(temp, (shape101, shape101), interpolation=cv2.INTER_CUBIC)
    if (i % 10) != 9:
        x_train.append(data)
    else:
        x_test.append(data)
x_train = np.array(x_train).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)

# 1.2 get target data
def readLabel(fileName):
    label = []
    f = open(fileName, "r")
    line = f.readline().rstrip("\n")
    while line != "":
        components = line.split(" ")
        label.append(int(components[1]))
        line = f.readline().rstrip("\n")
    f.close()
    return label
y_train = readLabel("ele/train.txt")
y_test = readLabel("ele/test.txt")



# set parameters
num_classes = 15
# read data
print('x_train shape:', x_train.shape)  # (3213, 101, 101, 3)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
x_test = x_test.astype('float32')
y_test = keras.utils.to_categorical(y_test, num_classes)


# 2. training model
batch_size = 128
epochs = 1000
# build model
# what if don't use padding?
def get_model(regularize_version, weight_decay=0.001):
    kernel_dim = 3
    if regularize_version == 0:
        model = Sequential()
        model.add(Conv2D(32, (kernel_dim, kernel_dim), padding='same',
                         input_shape=x_train.shape[1:], activation='relu', name='input'))
        # model.add(BatchNormalization())
        model.add(Conv2D(32, (kernel_dim, kernel_dim), padding='same',
                         input_shape=x_train.shape[1:], activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(32, (kernel_dim, kernel_dim), padding='same',
        #                  input_shape=x_train.shape[1:]))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(64, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(BatchNormalization())
        model.add(Conv2D(64, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, (kernel_dim, kernel_dim), padding='same'))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(Conv2D(128, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, (kernel_dim, kernel_dim), padding='same'))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(BatchNormalization())
        model.add(Conv2D(256, (kernel_dim, kernel_dim), padding='same', activation='relu'))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))  #512
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='output'))
    else:
        model = Sequential()
        model.add(Conv2D(32, (kernel_dim, kernel_dim), padding='same',
                         input_shape=x_train.shape[1:], activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay), name='input'))
        model.add(Conv2D(32, (kernel_dim, kernel_dim), padding='same',
                         input_shape=x_train.shape[1:], activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(64, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv2D(64, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv2D(128, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv2D(256, (kernel_dim, kernel_dim), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))  # 512
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay), name='output'))
    return model
weight_decay = 0.001
regularize_version = 1
model = get_model(regularize_version)
# opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)  # learning_rate
opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
train_history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=0.1,
                          shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



# 3. save model
model.save("demo1.h5")
model.save_weights('demo1_weights.h5')

saver = tf.train.Saver()  # put at front of file?
# saver = tf.train.Saver()  # put at front of file?
model2 = keras.models.load_model("demo1.h5")
# model2.load_weights('./weigths.h5',by_name=True)
scores = model2.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

sess = keras.backend.get_session()
save_path = saver.save(sess, "demo1.ckpt")





# 4. show pictures
# show the accuracy history
def show_accuracy(train_history, epochs):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title('Accuracy History')
    plt.ylabel('Accuracy Rate')
    plt.xlabel('Epoch')
    # plt.xticks(np.linspace(0, epochs, epochs + 1))
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()
# show learning curve
def show_learning_curve(train_history, epochs):
    plt.plot(train_history.history['val_loss'])
    plt.title('Learning Curve')
    plt.ylabel('VAL_Loss')
    plt.xlabel('Epoch')
    # plt.xticks(np.linspace(0, epochs, epochs + 1))
    plt.legend(['Cross Entropy', ], loc='upper right')
    plt.show()
show_accuracy(train_history, epochs)
show_learning_curve(train_history, epochs)



if __name__ is '__main__':
    parser = init_argparser()
    args, _ = parser.parse_known_args()
    exit(main(args))