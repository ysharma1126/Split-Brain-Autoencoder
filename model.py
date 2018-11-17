import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from skimage import color
from cifar import Cifar
from time import gmtime, strftime


###########################################################
# L : [0, 100], A: [-86.185, 98.254], B: [-107.863, 94.482]
###########################################################

def residual(net, num_filt, kernel_size, keepProb, isTraining, isFirst, isLast):
    with slim.arg_scope([slim.layers.convolution], 
        padding='SAME',
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=None
        ):

        if isFirst:
            net = tf.nn.relu(net)
        save = net
        net = slim.layers.batch_norm(net, is_training=isTraining)
        net = tf.nn.relu(net)
        net = slim.layers.convolution(net, num_filt, kernel_size, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
            activation_fn=None)
        net = slim.layers.dropout(net, keep_prob=keepProb, is_training=isTraining)
        net = slim.layers.batch_norm(net, is_training=isTraining)
        net = tf.nn.relu(net)
        net = slim.layers.convolution(net, num_filt, kernel_size, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
            activation_fn=None)
        net = save + net
        if isLast:
            net = tf.nn.relu(net)
        return net

#input_distortion cannot be used with classification because we could not perform
#the same transformation in numpy for pre-augmentation and quantization
def input_distortion(images, isTraining, batch_size):
    images = tf.cond(isTraining, lambda: tf.random_crop(images, [batch_size, 24, 24, 3]), 
        lambda: tf.map_fn(lambda image: tf.image.resize_image_with_crop_or_pad(image, target_height=24, target_width=24), images))
    images = tf.cond(isTraining, lambda: tf.map_fn(tf.image.random_flip_left_right, images), lambda: images)
    return images

class Model():

    def __init__(self, sess, data, val_data, test_data, cifar, num_iter, sup_learning_rate, uns_learning_rate_1, uns_learning_rate_2, batch_size, 
        test_size, is_supervised, is_untrained):
        self.sess = sess
        self.data = data #initialize this with Cifar.data
        self.val_data = val_data
        self.test_data = test_data
        self.cifar = cifar
        self.num_iter = num_iter
        self.sup_learning_rate = sup_learning_rate
        self.uns_learning_rate_1 = uns_learning_rate_1
        self.uns_learning_rate_2 = uns_learning_rate_2
        self.batch_size = batch_size
        self.test_size = test_size
        self.is_supervised = is_supervised
        self.sup_percentage = None
        self.is_untrained = is_untrained
        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.build_model(self.is_supervised)

    def change_sup_percentage(self, percentage):
        self.sup_percentage = percentage

    def build_model(self, is_supervised):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.isTraining = tf.placeholder(tf.bool, shape=[])
        
        if is_supervised:
            self.y = tf.placeholder(tf.float32, shape=[None, 10])
            result = self.unsupervised_arch(self.x)

            self.L_feature_map = result[2]
            self.ab_feature_map = result[3]

            if not self.is_untrained:
                self.saver = tf.train.Saver()

            self.prediction = self.supervised_arch(self.L_feature_map, self.ab_feature_map)
            tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.prediction)

            self.sup_loss = tf.losses.get_total_loss()
            self.reg_loss = tf.losses.get_regularization_losses()
            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=1), tf.argmax(input=self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.total_corr = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32))
            self.sup_saver = tf.train.Saver()

            if self.is_untrained:
                train_loss_sum = tf.summary.scalar('Supervised Untrained Training Loss', self.sup_loss)
                train_acc_sum = tf.summary.scalar('Supervised Untrained Training Accuracy', self.accuracy)

                val_loss_sum = tf.summary.scalar('Supervised Untrained Val Loss', self.sup_loss)
                val_acc_sum = tf.summary.scalar('Supervised Untrained Val Accuracy', self.accuracy)

                self.train_merged = tf.summary.merge([train_loss_sum, train_acc_sum])
                self.val_merged = tf.summary.merge([val_loss_sum, val_acc_sum])
                self.log_writer = tf.summary.FileWriter('./train_unt_sup_logs/'+self.time, self.sess.graph)                

            else:     
                train_loss_sum = tf.summary.scalar('Supervised Trained Training Loss', self.sup_loss)
                train_acc_sum = tf.summary.scalar('Supervised Trained Training Accuracy', self.accuracy)

                val_loss_sum = tf.summary.scalar('Supervised Trained Val Loss', self.sup_loss)
                val_acc_sum = tf.summary.scalar('Supervised Trained Val Accuracy', self.accuracy)

                self.train_merged = tf.summary.merge([train_loss_sum, train_acc_sum])
                self.val_merged = tf.summary.merge([val_loss_sum, val_acc_sum])
                self.log_writer = tf.summary.FileWriter('./train_tra_sup_logs/'+self.time, self.sess.graph)

        if not is_supervised:
            self.images = self.x
            result = self.unsupervised_arch(self.x)

            self.L_reg, self.ab_reg, self.L_hat_reg, self.ab_hat_reg = (x for x in result[0])

            L, ab, L_hat, ab_hat = result[0]

            #self.L_hat_maxed, and self.ab_hat_maxed used for displaying predicted image
            self.L_hat_maxed = tf.reshape(tf.argmax(L_hat, axis=3), [-1, 16, 16, 1])
            self.ab_hat_maxed = tf.reshape(tf.argmax(ab_hat, axis=3), [-1, 16, 16, 1])

            L_hat = tf.reshape(L_hat, [-1, 16*16, 100])
            ab_hat = tf.reshape(ab_hat, [-1, 16*16, 625])
            self.L = tf.placeholder(dtype=tf.int32, shape=[None, 16*16])
            self.ab = tf.placeholder(dtype=tf.int32, shape=[None, 16*16])

            self.L_labels_per_im = tf.unstack(self.L, num=self.batch_size)
            self.ab_labels_per_im = tf.unstack(self.ab, num=self.batch_size)
            self.L_hat_per_im = tf.unstack(L_hat, num=self.batch_size)
            self.ab_hat_per_im = tf.unstack(ab_hat, num=self.batch_size)

            self.ab_hat_loss = tf.zeros([])
            self.L_hat_loss = tf.zeros([])

            for index in range(len(self.L_labels_per_im)):
                self.ab_hat_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.ab_labels_per_im[index],
                    logits=self.ab_hat_per_im[index]))
                self.L_hat_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.L_labels_per_im[index],
                    logits=self.L_hat_per_im[index]))

            self.ab_hat_loss = self.ab_hat_loss / self.batch_size
            self.L_hat_loss = self.L_hat_loss / self.batch_size

            train_ab_sum = tf.summary.scalar('Unsupervised Training ab_hat loss', self.ab_hat_loss)
            train_l_sum = tf.summary.scalar('Unsupervised Training L hat loss', self.L_hat_loss)

            val_ab_sum = tf.summary.scalar('Unsupervised Val ab_hat loss', self.ab_hat_loss)
            val_l_sum = tf.summary.scalar('Unsupervised Val L hat loss', self.L_hat_loss)

            self.train_merged = tf.summary.merge([train_ab_sum, train_l_sum])
            self.val_merged = tf.summary.merge([val_ab_sum, val_l_sum])
            self.log_writer = tf.summary.FileWriter('./train_uns_logs/'+self.time, self.sess.graph)

            self.saver = tf.train.Saver()
    
    def supervised_arch(self, L_feature_map, ab_feature_map):
        self.total_features = tf.concat(
            [L_feature_map, ab_feature_map],
            axis=3
            )

        with tf.variable_scope('Supervised'):
            with slim. arg_scope([slim.layers.convolution, slim.layers.fully_connected],
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                #normalizer_fn = slim.layers.batch_norm,
                #normalizer_params = {'is_training': self.isTraining, 'updates_collections': ['supervised_update_coll'], 'scale': True},
                variables_collections = ['supervised_var_coll']
                ):
                #result = slim.layers.convolution(self.total_features, 32, [3, 3], scope='S_conv1') # 12 x 12 x 64
                #result = residual(result, 32, [3, 3], 0.5, self.isTraining, True, True)
                result = slim.layers.flatten(self.total_features)
                #result = slim.layers.fully_connected(result, 1024, weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
                #result = slim.layers.dropout(result, keep_prob=0.5, is_training=self.isTraining)
                result = slim.layers.fully_connected(result, 10, activation_fn=None, normalizer_fn=None)

        return result

    def unsupervised_arch(self, images):
        #images = input_distortion(images, self.isTraining, self.batch_size)
        L = tf.reshape(images[:, :, :, 0], shape=[-1, 32, 32, 1])
        ab = tf.concat(
            [tf.reshape(images[:, :, :, 1], shape=[-1, 32, 32, 1]), tf.reshape(images[:, :, :, 2], shape=[-1, 32, 32, 1])],
            axis=3
            )

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training': self.isTraining},
            variables_collections = ['unsupervised_ab_hat'],
            ):

            ab_hat = slim.layers.convolution(L, 64, [3, 3], scope='L_conv1')
            ab_hat = slim.layers.convolution(ab_hat, 64, [3, 3], scope='L_conv2')
            ab_hat = slim.layers.max_pool2d(ab_hat, [2, 2])
            ab_hat = slim.layers.convolution(ab_hat, 64, [3, 3], scope='L_conv3')

            ### PUT THIS LINE WHERE YOU WANT TO EXTRACT SUPERVISED AB FEATURES ###            
            #ab_features = ab_hat

            #with tf.variable_scope('L_res1'):
            #    ab_hat = residual(ab_hat, 64, [3, 3], 0.7, self.isTraining, True, False)
            
            ab_hat = slim.layers.convolution(ab_hat, 256, [3, 3], scope='L_conv4')

            ab_hat = slim.layers.convolution(ab_hat, 256, [3, 3], scope='L_conv5')

            #with tf.variable_scope('L_res2'):
            #    ab_hat = residual(ab_hat, 256, [3, 3], 0.7, self.isTraining, False, True) # 12 x 12 x 64

            ab_hat = slim.layers.convolution(ab_hat, 256, [3, 3], scope='L_conv6')

            ab_features = ab_hat
            ab_hat = slim.layers.convolution(ab_hat, 625, [1, 1], scope='L_conv7', activation_fn=None, normalizer_fn=None)
                #normalizer_params = {'is_training' : self.isTraining, 'scale' : True}) # 12 x 12 x 2

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training': self.isTraining}, 
            variables_collections = ['unsupervised_L_hat'],  
            ):

            L_hat = slim.layers.convolution(ab, 64, [3, 3], scope='ab_conv1') # 24 x 24 x 32
            L_hat = slim.layers.convolution(L_hat, 64, [3, 3], scope='ab_conv2')
            L_hat = slim.layers.max_pool2d(L_hat, [2, 2]) # 12 x 12 x 32
            L_hat = slim.layers.convolution(L_hat, 64, [3, 3], scope='ab_conv3') # 12 x 12 x 64

            ### PUT THIS LINE WHERE YOU WANT TO EXTRACT SUPERVISED L FEATURES ###
            #L_features = L_hat

            #with tf.variable_scope('ab_res1'):
            #    L_hat = residual(L_hat, 64, [3, 3], 0.7, self.isTraining, True, False) # 12 x 12 x 64

            L_hat = slim.layers.convolution(L_hat, 256, [3, 3], scope='ab_conv4')

            L_hat = slim.layers.convolution(L_hat, 256, [3, 3], scope='ab_conv5')

            #with tf.variable_scope('ab_res2'):
            #    L_hat = residual(L_hat, 256, [3, 3], 0.7, self.isTraining, False, True) # 12 x 12 x 64

            L_hat = slim.layers.convolution(L_hat, 256, [3, 3], scope='ab_conv6')

            # L_hat = slim.layers.convolution(L_hat, 256, [3, 3], scope='ab_conv6')
            L_features = L_hat
            L_hat = slim.layers.convolution(L_hat, 100, [1, 1], scope='ab_conv7', activation_fn=None, normalizer_fn=None)
                #normalizer_params = {'is_training' : self.isTraining, 'scale' : True}) # 12 x 12 x 1

        # if using regression use the bottom two lines
        L = tf.image.resize_bilinear(L, [16, 16])
        ab = tf.image.resize_bilinear(ab, [16, 16])

        return [(L, ab, L_hat, ab_hat), images, L_features, ab_features]

    def count_variables(self):
        variables1 = tf.get_collection('unsupervised_ab_hat')
        variables2 = tf.get_collection('unsupervised_L_hat')

        count = 0
        for variable in variables1:
            print(variable.get_shape().as_list())
            shape = variable.get_shape().as_list()
            local_count = 1
            for item in shape:
                local_count *= item
            count += local_count

        for variable in variables2:
            print(variable.get_shape().as_list())
            shape = variable.get_shape().as_list()
            local_count = 1
            for item in shape:
                local_count *= item
            count += local_count

        print(count)


    def train_init(self):
        if self.is_supervised:
            if self.is_untrained:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    self.optim = tf.group(updates,
                        tf.train.AdamOptimizer(
                            learning_rate=self.sup_learning_rate
                            )
                            .minimize(self.sup_loss)
                        )
                else:
                    self.optim = tf.train.AdamOptimizer(
                        learning_rate=self.sup_learning_rate,
                        ).minimize(self.sup_loss)
            else:
                update_ops = tf.get_collection('supervised_update_coll')
                model_variables = tf.get_collection('supervised_var_coll')
                if update_ops:
                    updates = tf.group(*update_ops)
                    self.optim = tf.group(updates,
                        tf.train.AdamOptimizer(
                            learning_rate=self.sup_learning_rate
                            )
                            .minimize(self.sup_loss, var_list=model_variables)
                        )
                else:
                    print("NO UPDATE")
                    self.optim = tf.train.AdamOptimizer(
                        learning_rate=self.sup_learning_rate,
                        ).minimize(self.sup_loss, var_list=model_variables)

        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            ab_coll = tf.get_collection('unsupervised_ab_hat')
            L_coll = tf.get_collection('unsupervised_L_hat')
            if update_ops:
                updates = tf.group(*update_ops)

                self.optim = tf.group(updates,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_1)
                        .minimize(self.ab_hat_loss, var_list=ab_coll)
                    ,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_2)
                        .minimize(self.L_hat_loss, var_list=L_coll)
                    )
            else:
                self.optim = tf.group(
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_1)
                        .minimize(self.ab_hat_loss, var_list=ab_coll)
                    ,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_2)
                        .minimize(self.L_hat_loss, var_list=L_coll)
                    )

        self.sess.run(tf.global_variables_initializer())

        if self.is_supervised and not self.is_untrained:
            self.saver.restore(self.sess, './saved_uns_model/model.ckpt')

    def train_iter(self, iteration, x, y=None, L_labels=None, ab_labels=None):
        if self.is_supervised:
            if y is None:
                raise ValueError("Must supply labels for supervised training")

            loss, reg_loss, _, accuracy, summary = self.sess.run(
                [self.sup_loss, self.reg_loss, self.optim, self.accuracy, self.train_merged],
                feed_dict = {self.x: x, self.y: y, self.isTraining: True}
                )
            
            if self.is_untrained:
                print('SUPUN  loss: {0}, reg_loss: {3} accuracy: {1}, ITERATION: {2}'.format(loss, accuracy, iteration, reg_loss))
            else:
                print('SUPTRA loss: {0}, reg_loss: {3} accuracy: {1}, ITERATION: {2}'.format(loss, accuracy, iteration, reg_loss))

            self.log_writer.add_summary(summary, iteration)
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")
            L_hat_maxed, ab_hat_maxed, L_reg_e, L_hat_reg_e, ab_reg_e, ab_hat_reg_e, ab_hat_loss, L_hat_loss, _, summary, ims = self.sess.run(
                [self.L_hat_maxed, self.ab_hat_maxed, self.L_reg, self.L_hat_reg, self.ab_reg, self.ab_hat_reg, self.ab_hat_loss, self.L_hat_loss, self.optim, self.train_merged, self.images], 
                feed_dict = {self.x: x, self.L: L_labels, self.ab: ab_labels, self.isTraining: True}
                )

            #if iteration % 1000 == 0:
                #print(self.cifar.denormalize_image(np.concatenate((L_reg_e[0], ab_reg_e[0]), axis=2)))
                #plt.subplot(1,2,1)
                #plt.imshow(color.lab2rgb(self.cifar.denormalize_image(np.concatenate((L_reg_e[0], ab_reg_e[0]), axis=2).astype(np.float64))))
                #plt.title('Real Image')
                #plt.imshow(color.lab2rgb(self.cifar.dequantize(np.concatenate((L_hat_maxed[0], ab_hat_maxed[0]), axis=2).astype(int)).astype(np.float64)))
                #plt.title('Colorized Image')
                #plt.tight_layout()
                #plt.savefig('train_'+str(iteration)+'.pdf', format='pdf', bbox_inches='tight')

            print('ab_hat_l2loss: {0}, L_hat_l2_loss: {1}, ITERATION: {2}'.format(ab_hat_loss, L_hat_loss, iteration))
            self.log_writer.add_summary(summary, iteration)

    def info_iter(self, iteration, x, y=None, L_labels=None, ab_labels=None):
        if self.is_supervised:
            if y is None:
                raise ValueError("Must supply labels for supervised training")

            loss, reg_loss, accuracy, summary = self.sess.run(
                [self.sup_loss, self.reg_loss, self.accuracy, self.val_merged],
                feed_dict = {self.x: x, self.y: y, self.isTraining: False}
                )

            print('SUP-- VAL: loss:{0}, reg_loss: {3}, accuracy: {1}, ITERATION: {2}'.format(loss, accuracy, iteration, reg_loss))
            self.log_writer.add_summary(summary, iteration)
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")

            ab_hat_loss, L_hat_loss, summary = self.sess.run(
                [self.ab_hat_loss, self.L_hat_loss, self.val_merged],
                feed_dict = {self.x: x, self.L: L_labels, self.ab: ab_labels, self.isTraining: False}
                #feed_dict = {self.x: x, self.isTraining: False}
                )
            print('VAL: ab_hat_l2_loss: {0}, L_hat_l2_loss: {1}, ITERATION: {2}'.format(ab_hat_loss, L_hat_loss, iteration))
            self.log_writer.add_summary(summary, iteration)

    def test(self):
        if not self.is_supervised:
            for idx, (x, L_labels, ab_labels) in enumerate(self.test_data(self.test_size, self.is_supervised)):
                L_hat_maxed, ab_hat_maxed, L_reg_e, L_hat_reg_e, ab_reg_e, ab_hat_reg_e, ab_hat_loss, l_hat_loss = self.sess.run(
                    [self.L_hat_maxed, self.ab_hat_maxed, self.L_reg, self.L_hat_reg, self.ab_reg, self.ab_hat_reg, self.ab_hat_loss, self.L_hat_loss],
                    feed_dict={self.x : x, self.L: L_labels, self.ab: ab_labels, self.isTraining: False},
                    )
                print("TEST AB LOSS: {0}, TEST L LOSS : {1}".format(ab_hat_loss, l_hat_loss))
                #if idx == 1000:
                    #plt.subplot(1,2,1)
                    #plt.imshow(color.lab2rgb(self.cifar.denormalize_image(np.concatenate((L_reg_e[0], ab_reg_e[0]), axis=2).astype(np.float64))))
                    #plt.title('Real Image')
                    #plt.subplot(1,2,2)
                    #plt.imshow(color.lab2rgb(self.cifar.dequantize(np.concatenate((L_hat_maxed[0], ab_hat_maxed[0]), axis=2).astype(int)).astype(np.float64)))
                    #plt.title('Colorized Image')
                    #plt.tight_layout()
                    #plt.savefig('test_'+str(idx)+'.pdf', format='pdf', bbox_inches='tight')

        if self.is_supervised:
            total_corr = 0
            for x, y in self.test_data(self.test_size, self.is_supervised):
                total_corr += self.sess.run(
                    self.total_corr,
                    feed_dict={self.x: x, self.y: y, self.isTraining: False}
                    )
                print(total_corr)
            print("TEST ACCURACY: {0}".format(total_corr*100/10000))

    def train(self):
        for iteration in range(self.num_iter):
            if self.is_supervised:
                x, y= self.data(self.batch_size, self.is_supervised, self.sup_percentage)
                self.train_iter(iteration, x, y)

                if iteration % 1000 == 0:
                    self.info_iter(iteration, x, y)

                # if iteration % 1000 == 0:
                #     self.test()

            else:
                x, L_labels, ab_labels = self.data(self.batch_size, self.is_supervised)

                self.train_iter(iteration, x, L_labels=L_labels, ab_labels=ab_labels)

                if iteration % 1000 == 0:
                    x, L_labels, ab_labels = self.val_data(self.batch_size, self.is_supervised)
                    self.info_iter(iteration, x, L_labels=L_labels, ab_labels=ab_labels)

        if not self.is_supervised:
            save_path = self.saver.save(self.sess, "./saved_uns_model/model.ckpt")
            print("Unsupervised Model saved in file: %s" % save_path)
        else:
            if self.is_untrained:
                save_path = self.sup_saver.save(self.sess, "./saved_sup_unt_model/model.ckpt")
                print("Supervised Untrained Model saved in file: %s" % save_path)
            else:
                save_path = self.sup_saver.save(self.sess, "./saved_sup_tra_model/model.ckpt")
                print("Supervised Trained Model saved in file: %s" % save_path)
