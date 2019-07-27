'''
Model trainer.
Author: Rodrigo de la Iglesia.
Version: 1.0.
27/07/2019
'''

import tensorflow as tf
import sys
import os
import datetime

from input_pipeline import Pipeline
from create_model import Model
from helper_func import *

from sklearn import preprocessing

one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)

flags = tf.app.flags
FLAGS = flags.FLAGS
#Training parameters.
flags.DEFINE_string('base_path', '', """Directory where training images are stored.""")
flags.DEFINE_string('save_dir', '', """Directory where trained model and variables will be stored.""")
flags.DEFINE_float('valid_size', 0.2, """Validation dataset size.""")
flags.DEFINE_integer('n_epochs', 100, """Training epochs number.""")
flags.DEFINE_integer('batch_size_train', 1, """Training batch size.""")
flags.DEFINE_integer('batch_size_valid', 1, """Validation batch size.""")
flags.DEFINE_float('learning_rate', 0.0001, """Gradient descent learning rate.""") 
flags.DEFINE_integer('image_h', 1100, """Processed images height.""")
flags.DEFINE_integer('image_w', 1024, """Processed images width.""")


def main(argv):
    date = datetime.datetime.now()

    #Dataset object.
    dataset = Pipeline(FLAGS.base_path, FLAGS.image_h, FLAGS.image_w)

    handle = dataset.handle
    #Load data lists.
    train_x, train_y, train_n, valid_x, valid_y, valid_n = dataset.createList(valid_size=0.2)

    #Datasets and iterator creation.
    dataset_train = dataset.createDataset(train_x, train_y, train_n, FLAGS.batch_size_train)
    train_iterator = dataset.initializeIterator(dataset_train, one_shot=False)
    dataset_valid = dataset.createDataset(valid_x, valid_y, valid_n, FLAGS.batch_size_valid)
    valid_iterator = dataset.initializeIterator(dataset_valid, one_shot=False)

    #Train data returned by iterator.
    batch = dataset.createIterator(dataset_train)

    #Object model.
    model = Model(dataset.n_classes, batch[0], batch[1], FLAGS.learning_rate) 
    save_dir = FLAGS.save_dir
    #Saver object.
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    os.makedirs(save_dir+'/'+date.strftime('%y_%m_%d-%H_%M'))
    save_dir = save_dir+'/'+date.strftime('%y_%m_%d-%H_%M')
    save_path = os.path.join(save_dir, 'best_validation')

    #Steps number for training and validation. 
    n_steps_train = int(len(train_x)/FLAGS.batch_size_train)
    n_steps_valid = int(len(valid_x)/FLAGS.batch_size_valid)

    #Initialize Tensorflow session.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #Handle: Decide which dataset (train or valid) is loaded in each operation.
        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        v_loss_train = []
        v_loss_valid = []
        v_acc_train = []
        v_acc_valid = []

        #Early stopping parameters.
        #Best validation accuracy obtained.
        best_validation_accuracy = 0.0
        #Last epoch where validation accuracy improved.
        last_improvement = 0
        #Max. epoch number without improvement. Once is reached, training process will stop.
        #Número de épocas a las que el entrenamiento es detenido si no ha habido mejora.
        improvement_epochs = 10

        for epoch in range(FLAGS.n_epochs):
            #Train model for one epoch.
            print("\nTraining...")
            sess.run(train_iterator.initializer)
            sum_loss_train = 0
            sum_acc_train = 0
            i = 0

            while True:
                try:
                    _, loss_train, acc_train = sess.run([model.optimizer, model.loss, model.accuracy],
                                                        feed_dict={handle:train_handle, model.keep_prob:0.5})

                    sum_loss_train += loss_train
                    sum_acc_train += acc_train

                    showProgress(epoch, i, n_steps_train, loss_train, acc_train)             
                    checkRAM()
                    i += 1

                except tf.errors.OutOfRangeError:
                    mean_loss_train = sum_loss_train/n_steps_train
                    mean_acc_train = sum_acc_train/n_steps_train
                    v_loss_train.append(mean_loss_train)
                    v_acc_train.append(mean_acc_train)
                    
                    showEpochResults(mean_loss_train, mean_acc_train)
                    break            

            sess.run(valid_iterator.initializer)

            #Validate model for one epoch.
            print("\nValidating...")
            sum_loss_valid = 0
            sum_acc_valid = 0
            j = 0

            while True:
                try:
                    loss_valid, acc_valid = sess.run([model.loss, model.accuracy],
                                                    feed_dict={handle:valid_handle, model.keep_prob:1})

                    sum_loss_valid += loss_valid
                    sum_acc_valid += acc_valid

                    showProgress(epoch, j, n_steps_valid, loss_valid, acc_valid)                
                    checkRAM()
                    j += 1

                except tf.errors.OutOfRangeError:
                    mean_loss_valid = sum_loss_valid/n_steps_valid
                    mean_acc_valid = sum_acc_valid/n_steps_valid
                    v_loss_valid.append(mean_loss_valid)
                    v_acc_valid.append(mean_acc_valid)
                    
                    showEpochResults(mean_loss_valid, mean_acc_valid)
                    break

            #If validation accuracy increased in last epoch.
            if mean_acc_valid > best_validation_accuracy:
                #Update best accuracy value.
                best_validation_accuracy = mean_acc_valid
                last_improvement = epoch

                #Save trained variables.
                saver.save(sess=sess, save_path=save_path)
                print('Improvement')

            #If there wasn't improvements in a while, stop training.
            if epoch - last_improvement > improvement_epochs:
                print('No improvements in a while. Stopping optimization.')
                break
                
        #Write training data in text file and save it.
        f = open(save_dir+'/parameters.txt', 'w')
        f.write('Data set:\t{}\nClasses:\t{}\nValidation set size:\t{}\nEpochs number:\t{}\nBathch size train:\t{}\nBathch size validation:\t{}\nLearning rate:\t{}\nImage size:\t{},{}\nBest validation accuracy:\t{}'.format(
            FLAGS.base_path,
            str(dataset.classes),
            str(FLAGS.valid_size),
            str(FLAGS.n_epochs),
            str(FLAGS.batch_size_train),
            str(FLAGS.batch_size_valid),
            str(FLAGS.learning_rate),
            str(FLAGS.image_h),
            str(FLAGS.image_w),
            str(best_validation_accuracy)))
        f.close()

        #Plot training results.    
        plotResults(1, v_loss_train, v_loss_valid, loss=True, title='Train and validation loss.')
        plotResults(2, v_acc_train, v_acc_valid, loss=False, title='Train and validation accuracy')


if __name__ == "__main__":
    tf.app.run()