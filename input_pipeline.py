'''
Immage input pipelines.
Author: Rodrigo de la Iglesia.
Version: 1.0.
27/07/2019
'''
import tensorflow as tf
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split

def dataGenerator(data_x, data_y, data_n):
    #Data generator: image direction in memory, label and class name.
    for x, y, n in zip(data_x, data_y, data_n):
        yield x, y, n

class Pipeline:
    '''
    Class Pipeline: Methods to create input pipelines using generators.
    Yield train data to training algorithm preventing memory overflow. 
    '''
    def __init__(self, base_path, image_h, image_w):
        self.base_path = base_path
        self.image_h = image_h
        self.image_w = image_w
        self.classes = os.listdir(self.base_path)
        self.n_classes = len(self.classes)
        self.handle = tf.placeholder(tf.string, shape=[])


    def processData(self, path, label, name):
        #Read and load images from directions.
        #Process image loaded.
        img = tf.image.decode_bmp(tf.read_file(path))
        img = img[:,:,:3] #Remove alpha channel(opacity)
        img.set_shape([None, None, 3])
        img = tf.image.resize_images(img,[self.image_h, self.image_w])
        #Normalize images.
        img = tf.to_float(img) * (1/255.)
        return img, label, name
        
    def createList(self, valid_size=None):
        #Create image directions an labels lists.
        paths = []
        labels = []
        one_hot_labels = []
               
        for folder in self.classes:
            folder_idx = self.classes.index(folder)
            for i in glob.glob(os.path.join(self.base_path, folder, '*.bmp')):
                label = i.split(os.path.sep)[-2]
                one_hot_label = np.zeros(self.n_classes)
                one_hot_label[folder_idx] = 1.0
                paths.append(i)
                labels.append(label)
                one_hot_labels.append(one_hot_label)

        #One hot labels.
        labels = np.array(labels)
        labels = labels.reshape(len(labels), 1)
        one_hot_labels = np.array(one_hot_labels)
        #Shuffle list elemets.
        list_combined = sorted(list(zip(paths, one_hot_labels, labels)))
        np.random.shuffle(list_combined)
        paths, one_hot_labels, labels = zip(*list_combined)

        if valid_size is not None:
            #If validation dataset is needed.
            (train_x, valid_x, train_y, valid_y, train_n, valid_n) = train_test_split(paths,
                                                                                      one_hot_labels,
                                                                                      labels,
                                                                                      test_size=valid_size,
                                                                                      random_state=42)
            return train_x, train_y, train_n, valid_x, valid_y, valid_n

        return paths, one_hot_labels, labels

        
    def createDataset(self, data_x, data_y, data_n, batch_size):
        #Create dataset.
        dataset = tf.data.Dataset.from_generator(lambda: dataGenerator(data_x, data_y, data_n), (tf.string, tf.float32, tf.string))
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(self.processData, num_parallel_calls=2)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)
        
        return dataset
    
    def createIterator(self, dataset):
        #Iterator object.
        iterator = tf.data.Iterator.from_string_handle(self.handle, dataset.output_types, dataset.output_shapes)
        #Elements returned from iterator.
        batch = iterator.get_next()
            
        return batch
    
    def initializeIterator(self, dataset, one_shot=True):
        #Iterator initializer.
        if one_shot:
            init_op = dataset.make_one_shot_iterator()
            
        init_op = dataset.make_initializable_iterator()
        
        return init_op