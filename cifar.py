import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color
from skimage import transform

import pickle
import os

np.random.seed(31415)

class Cifar():

    def one_hot(labels, size):
        """
        Create one-hot encodings for each of the class labels
        """
        a = np.zeros((len(labels), size), 'uint8')
        for ind in range(len(labels)):
            a[ind][labels[ind]] = 1
        return a

    def unpickle(file):
        """
        Unpickle the CIFAR-10 file
        """
        fo = open(file, 'rb')
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
        return dict

    def _convert_images(raw, num_channels, dim_size):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, num_channels, dim_size, dim_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])
        return images

    def __init__(self, path):
        train_data_batches = [Cifar.unpickle(path +'/data_batch_'+str(i)) for i in range(1, 6)]
        test_data_batch = Cifar.unpickle(path +'/test_batch')

        self.val_images = Cifar._convert_images(train_data_batches[0][b'data'], 3, 32)
        self.val_labels = np.array(train_data_batches[0][b'labels'])

        self.train_images = Cifar._convert_images(train_data_batches[1][b'data'], 3, 32)
        self.train_labels = np.array(train_data_batches[1][b'labels'])
        for i in range(2, 5):
            self.train_images = np.append(self.train_images, Cifar._convert_images(train_data_batches[i][b'data'], 3, 32), axis=0)
            self.train_labels = np.append(self.train_labels, np.array(train_data_batches[i][b'labels']), axis=0)

        self.test_images = Cifar._convert_images(test_data_batch[b'data'], 3, 32)
        self.test_labels = np.array(test_data_batch[b'labels'])

        #print(self.test_images.shape)

    def compute_maxs(self, lab_images):
        self.l_max = np.amax(lab_images[:, :, :, 0])
        self.a_max = np.amax(lab_images[:, :, :, 1])
        self.b_max = np.amax(lab_images[:, :, :, 2])
        self.a_min = np.abs(np.amin(lab_images[:, :, :, 1]))
        self.b_min = np.abs(np.amin(lab_images[:, :, :, 2]))
        print("MAX: {0}, MIN: {1}".format(self.a_max, self.a_min))
        print("MAX: {0}, MIN: {1}".format(self.b_max, self.b_min))

    def normalize(self, lab_images):
        lab_images[:, :, :, 0] = lab_images[:, :, :, 0] / self.l_max
        lab_images[:, :, :, 1] = (lab_images[:, :, :, 1] + self.a_min) / (self.a_max + self.a_min)
        lab_images[:, :, :, 2] = (lab_images[:, :, :, 2] + self.b_min) / (self.b_max + self.b_min)
        return lab_images

    def denormalize_image(self, lab_image):
        lab_image[:, :, 0] = lab_image[:, :, 0] * self.l_max
        lab_image[:, :, 1] = lab_image[:, :, 1] * (self.a_max + self.a_min) - self.a_min
        lab_image[:, :, 2] = lab_image[:, :, 2] * (self.b_max + self.b_min) - self.b_min
        return lab_image

    def quantize(lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 101, 101)) - 1
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-87, 99, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 16 + lab_images[:, :, :, 2]
        print("L_LABELS: {0}".format(l_labels.shape))
        print("AB_LABLES: {0}".format(ab_labels.shape))
        return l_labels.reshape([-1, 16*16]), ab_labels.reshape([-1, 16*16])

    def dequantize(self, lab_images):
        print(lab_images.shape)
        #print(lab_images)
        l_vals = np.linspace(0, 100, 100)
        print(l_vals.shape)
        a_vals = np.linspace(-87, 99.1, 16)
        b_vals = np.linspace(-108, 95.1, 16)
        a = (lab_images[:, :, 1] / 16).astype(int)
        b = (lab_images[:, :, 1] % 16).astype(int)
        print(lab_images[:, :, 0])
        lab_images[:, :, 0] = l_vals[lab_images[:, :, 0]]
        lab_images[:, :, 1] = a_vals[a]
        concat = b_vals[b].reshape((16, 16, 1))
        print(lab_images.shape)
        print(concat.shape)
        result = np.concatenate((lab_images, concat), axis=2)
        print(result)
        return result

    def quantize_rgb(lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 1.01, 101)) - 1
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(0, 1.01, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(0, 1.01, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 16 + lab_images[:, :, :, 2]
        print("L_LABELS: {0}".format(l_labels.shape))
        print("AB_LABLES: {0}".format(ab_labels.shape))
        return l_labels.reshape([-1, 16*16]), ab_labels.reshape([-1, 16*16])  

    def convert_to_lab(self):
        self.val_images = color.rgb2lab(self.val_images)
        self.train_images = color.rgb2lab(self.train_images)
        np.random.shuffle(self.train_images)
        self.test_images = color.rgb2lab(self.test_images)

        self.compute_maxs(self.train_images)
        #print(self.val_images[0])

        self.quantized_val_images = np.array([transform.resize(image, (16, 16), preserve_range=True) for image in self.val_images])
        self.quantized_train_images = np.array([transform.resize(image, (16, 16), preserve_range=True) for image in self.train_images])
        self.quantized_test_images = np.array([transform.resize(image, (16, 16), preserve_range=True) for image in self.test_images])

        # self.quantized_val_images = Cifar.quantize(self.val_images)
        self.train_l_labels, self.train_ab_labels = Cifar.quantize(self.quantized_train_images)
        self.val_l_labels, self.val_ab_labels = Cifar.quantize(self.quantized_val_images)
        self.test_l_labels, self.test_ab_labels = Cifar.quantize(self.quantized_test_images)

        # self.quantized_test_images = Cifar.quantize(self.test_images)
        #print(self.quantized_val_images[0])

        self.val_images = self.normalize(self.val_images)
        self.train_images = self.normalize(self.train_images)
        self.test_images = self.normalize(self.test_images)
        #print(self.val_images[0]) 

    def data(self, batch_size, is_supervised, percentage=None):
        if is_supervised:
            if not percentage is None:
                length = len(self.train_images) * (percentage/100)
                length = int(length)
            else:
                length = len(self.train_images)
            indices = np.random.randint(0, length, size=batch_size)
            x = self.train_images[indices]
            y = self.train_labels[indices]
            y = Cifar.one_hot(y, 10)
            return x, y

        else:
            indices = np.random.randint(0, len(self.train_images), size=batch_size)
            x = self.train_images[indices]
            l_labels = self.train_l_labels[indices]
            ab_labels = self.train_ab_labels[indices]
            #plt.imshow(np.squeeze(x[0]))
            #plt.show()
            #return x
            return x, l_labels, ab_labels

    def val_data(self, batch_size, is_supervised):
        if is_supervised:
            indices = np.random.randint(0, len(self.val_images), size=batch_size)
            x = self.val_images[indices]
            y = self.val_labels[indices]
            y = Cifar.one_hot(y, 10)
            return x,y

        else:
            indices = np.random.randint(0, len(self.val_images), size=batch_size)
            x = self.val_images[indices]
            l_labels = self.val_l_labels[indices]
            ab_labels = self.val_ab_labels[indices]
            #plt.imshow(np.squeeze(x[0]))
            #return x
            return x, l_labels, ab_labels

    def test_data(self, test_size, is_supervised):
        start = 0
        print(len(self.test_images))
        cont = True
        while cont:
            end = start + test_size
            if end >= len(self.test_images):
                end = len(self.test_images) - 1
                cont = False
            print("start: {0}, end: {1}".format(start, end))
            x = self.test_images[start:end]
            y = self.test_labels[start:end]
            l_labels = self.test_l_labels[start:end]
            ab_labels = self.test_ab_labels[start:end]
            y = Cifar.one_hot(y, 10)

            if is_supervised:
                yield x, y
            else:
                #yield x
                yield x, l_labels, ab_labels

            start = end