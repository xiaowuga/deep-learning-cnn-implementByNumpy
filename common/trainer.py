import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.optimier import *

class Trainer:
    '''
        进行神经网络的训练的类
    '''
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=0, batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        '''

        :param network: 网络结构
        :param x_train: 训练数据
        :param t_train: 训练数据的标签数据
        :param x_test:  测试数据
        :param t_test:  测试数据的标签数据
        :param epochs:  epoch
        :param batch_size: 批大小
        :param optimizer:  优化函数
        :param optimizer_param: 优化函数所需的超参数
        :param evaluate_sample_num_per_epoch:每个epoch结束后计算模型在训练集与测试集上精度时要测试的样本数量，默认为测试全部样本
        :param verbose: 是否输出log
        '''
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        # evaluate_sample_num_per_epoch为每个epoch结束后计算模型在训练集与测试集上精度时要测试的样本数量，默认为测试全部样本
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}

        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / batch_size, 1)
        self.max_iter = epochs * self.iter_per_epoch
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_iter += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))