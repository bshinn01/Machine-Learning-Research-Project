import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# my code starts here

# define cnn model
def define_cnn_model():
    cnn_Model = Sequential()
    cnn_Model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    cnn_Model.add(MaxPooling2D((2, 2)))
    cnn_Model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    cnn_Model.add(MaxPooling2D((2, 2)))
    cnn_Model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    cnn_Model.add(MaxPooling2D((2, 2)))
    cnn_Model.add(Flatten())
    cnn_Model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    cnn_Model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    cnn_Model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_Model


# plot the accuracy curve
def accuracy(history):
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('CNN: Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.show()


# define the test
def test():
    # define model
    cnn_model = define_cnn_model()
    # create data generators
    train_gen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    trainer = train_gen.flow_from_directory('dogs-vs-cats/CNN_subset_train/', class_mode='binary', batch_size=64, target_size=(200, 200))
    tester = test_gen.flow_from_directory('dogs-vs-cats/CNN_subset_val/', class_mode='binary', batch_size=64, target_size=(200, 200))
    # fit model
    history = cnn_model.fit_generator(trainer, steps_per_epoch=len(trainer),
                                  validation_data=tester, validation_steps=len(tester), epochs=50, verbose=0)
    # evaluate model
    _, acc = cnn_model.evaluate_generator(tester, steps=len(tester), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    accuracy(history)


# run test
test()

# my code ends here
