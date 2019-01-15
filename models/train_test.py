import sys
sys.path.insert(0, '../data/')
sys.path.insert(0, '../utils/')

import tensorflow as tf
import numpy as np

import layers
import data_loader
import os

class model(object):
    
    def __init__(self, x, is_training, hid_size, num_hid):
        """
            Args:
        """
        _y = layers.ser_fc_layers(
                inp=x,
                is_training=is_training,
                num_hid=num_hid,
                hid_dim=hid_size,
                out_dim=1,
                relu_after=False,
                add_bn=False,
                name="output",
                reuse=None)
        self._y = tf.squeeze(_y, axis=1)
        
        batchnorm_updates = tf.get_collection(
            layers.UPDATE_OPS_COLLECTION)
        self.batchnorm_updates_op = tf.group(*batchnorm_updates)


class SupervisedModel(object):
    """ Training a supervised model
    """

    def __init__(self,
                 model,
                 loader,
                 result_dir,
                 hid_size,
                 num_hid,
                 init_learning_rate=1e-3,
                 l2_reg=0.1,
                 max_grad_norm=0.5):
        """
        Assumption: access to a direct loader. the batch size directly decided
            in the loader
        """
        # initialization
        self.loader = loader
        self.result_dir = result_dir
        self.init_learning_rate = init_learning_rate
        self.hid_size = hid_size
        self.num_hid = num_hid
        self.l2_reg = l2_reg
        self.max_grad_norm = max_grad_norm

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.exists(self.result_dir.format('/train')):
            os.mkdir(self.result_dir.format('/train'))
        if not os.path.exists(self.result_dir.format('/validation')):
            os.mkdir(self.result_dir.format('/validation'))
        if not os.path.exists(self.result_dir.format('/test')):
            os.mkdir(self.result_dir.format('/test'))
        if not os.path.exists(self.result_dir.format('/model')):
            os.mkdir(self.result_dir.format('/model'))

        self.graph = tf.Graph()
        with self.graph.as_default():
            # session
            self.sess = tf.Session(graph=self.graph)

            # placeholders
            self.x = tf.placeholder(
                tf.float32, shape=[None, self.loader.bin_size])
            self.is_training = tf.placeholder(tf.bool, shape=[])
            self.y = tf.placeholder(tf.float32, shape=[None])
            self.learning_rate = tf.placeholder(tf.float32, shape=[])

            # model
            self.model = model(
                x=self.x,
                is_training=self.is_training,
                hid_size=self.hid_size,
                num_hid=self.num_hid)

            # losses
            self.l2_loss = 2 * tf.nn.l2_loss(self.y - self.model._y)
            loss_reg = 0
            for v in tf.trainable_variables():
                if not 'bias' in v.name.lower():
                    loss_reg += tf.nn.l2_loss(v)
            loss = self.l2_loss + (self.l2_reg * loss_reg)
            self.loss = loss
            
            # setting up the optmimizer
            self.params = tf.trainable_variables()
            grads = tf.gradients(loss, self.params)
            if self.max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(
                    grads, self.max_grad_norm)
            grads = list(zip(grads, self.params))

            trainer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, epsilon=1e-3)
    
            # applying the batchnorm update ops with training
            self._train = tf.group(self.model.batchnorm_updates_op,
                                   trainer.apply_gradients(grads))

            # tensorboard summaries
            tf.summary.scalar('l2_loss', self.l2_loss, collections=['tr'])
            tf.summary.scalar('loss_reg', loss_reg, collections=['tr'])
            tf.summary.scalar('loss', loss, collections=['tr'])

            # placeholders for logging
            self.comp_rmse = tf.placeholder(tf.float32, shape=[])

            tf.summary.scalar(
                "comp_rmse", self.comp_rmse, collections=['tr_2', 'va', 'te'])

            self.merged_tr = tf.summary.merge_all('tr')
            self.merged_tr_2 = tf.summary.merge_all('tr_2')
            self.merged_te = tf.summary.merge_all('te')
            self.merged_va = tf.summary.merge_all('va')

            # graph initialization
            init_op1 = tf.global_variables_initializer()
            self.sess.run(init_op1)
            init_opt2 = tf.local_variables_initializer()
            self.sess.run(init_opt2)

            # setting up tensorboard writers
            self.tr_writer = tf.summary.FileWriter(result_dir + '/train',
                                                   self.sess.graph)
            self.va_writer = tf.summary.FileWriter(result_dir + '/validation')
            self.te_writer = tf.summary.FileWriter(result_dir + '/test')
            self.saver = tf.train.Saver()

        self.steps_taken = 0
        # a value divisible by 2400
        # 2400 is the number of 'te' and 'va' samples
        self.log_batch_size = 2400
        self.va_loader = data_loader.loader(
            shuffle=False,
            bin_size=self.loader.bin_size,
            batch_size=self.log_batch_size,
            split='va',
            mu=self.loader.mu,
            sigma=self.loader.sigma)
        self.te_loader = data_loader.loader(
            shuffle=False,
            bin_size=self.loader.bin_size,
            batch_size=self.log_batch_size,
            split='te',
            mu=self.loader.mu,
            sigma=self.loader.sigma)

    def learn(self, log_loss=10, use_decay=False):
        self.steps_taken += 1
        x, y = self.loader.load()        
        feed_dict = {self.x: x, self.y: y, self.is_training: True}

        if use_decay:
            feed_dict[self.learning_rate] = (
                self.init_learning_rate / np.sqrt(self.steps_taken))
        else:
            feed_dict[self.learning_rate] = (self.init_learning_rate)
        
        summary, _, _y, l2_loss = self.sess.run((self.merged_tr, self._train, self.model._y, self.l2_loss),
                                   feed_dict=feed_dict)
        if self.steps_taken % log_loss == 0:
            self.tr_writer.add_summary(summary, self.steps_taken)

    def test_and_log(self, split):
        """ 
            Args:
                split (str): either va' or 'te' 
        """

        if split == 'va':
            writer = self.va_writer
            merged = self.merged_va
            loader = self.va_loader
        elif split == 'te':
            writer = self.te_writer
            merged = self.merged_te
            loader = self.te_loader
        
        total_loss = 0
        for i in range(loader.num_batches):
            x, y = loader.load()
            total_loss += self.sess.run(
                self.l2_loss,
                feed_dict={
                    self.x: x,
                    self.y: y,
                    self.is_training: False
                })

        rmse = np.sqrt(total_loss / (loader.num_batches * self.log_batch_size))
        return rmse
        
    def save(self):
        self.saver.save(self.sess, "{}/{}".format(self.result_dir, 'model'))
    
    def restore(self):
        self.saver.restore(self.sess, "{}/{}".format(self.result_dir, 'model'))

    def y_value(self, x_input):
        """ Returns the predicted y for x_input
        """

        feed_dict = {self.x: x_input, self.is_training: False}
        return self.sess.run([self._y], feed_dict)


train_size = 9600
batch_size = 9600
log_per_epoch = 10
early_stopping = 1000
def evaluate(bin_size, hid_size, num_hid, run):
    loader = data_loader.loader(
        shuffle=False,
        bin_size=bin_size,
        batch_size=batch_size,
        split='tr')
    network = SupervisedModel(
        model=model,
        loader=loader,
        result_dir='./{}_{}_{}_{}'.format(bin_size, hid_size, num_hid, run),
        num_hid=num_hid,
        hid_size=hid_size)

    batch_per_epoch = train_size // batch_size
    best_val_rmse = 100
    best_val_epoch = 0
    
    for epoch_i in range(100000):
        
        if epoch_i > (best_val_epoch + early_stopping):
            break
        
        if epoch_i % log_per_epoch == 0:
            # validation
            cur_val_rmse = network.test_and_log('va')
            if cur_val_rmse < best_val_rmse:
                best_val_rmse = cur_val_rmse
                best_val_epoch = epoch_i
                network.save()
        
        for batch_i in range(batch_per_epoch):
            network.learn(use_decay=True)
    
    network.restore()
    return (network.test_and_log('va'), network.test_and_log('te'))

exp1 = [{
    'bin_size': 20,
    'hid_size': hid_size,
    'num_hid': 5
} for hid_size in [10, 20, 40, 60, 80, 100]]
exp2 = [{
    'bin_size': bin_size,
    'hid_size': 20,
    'num_hid': 5
} for bin_size in [10, 20, 40, 60, 80, 100]]
exp3 = [{
    'bin_size': 20,
    'hid_size': 20,
    'num_hid': num_hid
} for num_hid in [1, 2, 5, 8, 10]]
for exp in exp1 + exp2 + exp3:
    for run in range(5):
        print("bin_size: {}, hid_size: {}, num_hid: {}, run: {}".format(
            exp['bin_size'], exp['hid_size'], exp['num_hid'], run))
        print(evaluate(exp['bin_size'], exp['hid_size'], exp['num_hid'], run))

