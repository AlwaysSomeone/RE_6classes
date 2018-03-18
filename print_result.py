import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score
FLAGS = tf.app.flags.FLAGS

def main(_):
    pathname = "./model/ATT_GRU_model-"
    wordembedding = np.load('./data/vec.npy')
    test_settings = network.Settings()
    #词向量大小规模
    test_settings.vocab_size = 16693
    test_settings.num_classes = 7
    test_settings.big_num = 1
    #为什么设为5561???
    #big_num = 1 时，PR curve area:0.517162123893
    #big_num = 10时，PR curve area:0.517162123893
    big_num_test = test_settings.big_num

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)
            saver = tf.train.Saver()

            testlist = [200]
            for model_iter in testlist:
                #读取训练9000次后的模型
                saver.restore(sess, pathname + str(model_iter))
                test_y = np.load('./data/testall_y.npy')
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')
                allprob = []
                acc = []
                #测试数据按bignum，一个big_num即是一个word_batch，分批喂入test_step
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    #增加到acc列表
                    #print(type(accuracy))
                    #print(accuracy)
                    #big_num = 10时，输出10次<class 'list'>
                    #accuracy是形如[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]的输出
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    #每big_num次测试的平均准确率增加到acc列表中去
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                order = np.argsort(-allprob)
                print(order)
                current_step = model_iter

                #print(acc)
                print(str(model_iter) + '次训练结果')
                print('准确率' + str(sum(acc)/len(acc)))
                print('saving all test result...')
                np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
                allans = np.load('./data/allans.npy')

                # caculate the pr curve area
                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))

if __name__ == "__main__":
	tf.app.run()