import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from tensorflow.contrib.tensorboard.plugins import projector


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):

    starttime = datetime.datetime.now().isoformat()
    print('main函数开始执行' + starttime)

    # the path to save models
    save_path = './model/'

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/train_y.npy')
    train_word = np.load('./data/train_word.npy')
    train_pos1 = np.load('./data/train_pos1.npy')
    train_pos2 = np.load('./data/train_pos2.npy')

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num
    #你一旦开始你的任务，就已经有一个默认的图已经创建好了。
    # 而且可以通过调用tf.get_default_graph()来访问到。
    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            #训练以及加载数据并不费时，费时的是加载网络
            initializer = tf.contrib.layers.xavier_initializer()
            # 返回一个用于定义创建variable（层）的op的上下文管理器。
            time_str1 = datetime.datetime.now().isoformat()
            print(time_str1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            time_str2 = datetime.datetime.now().isoformat()
            print(time_str2)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)

            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            #我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，
            #以便下次迭代的训练或者用作测试。Tensorflow针对这一需求提供了Saver类。
            saver = tf.train.Saver(max_to_keep=None)

            #为了使用Tensorboard来可视化我们的数据，
            #我们会经常使用Summary，最终都会用一个简单的merge_all函数来管理我们的Summary
            #状态可视化,为了释放TensorBoard所使用的事件文件（events file），
            #所有的即时数据（在这里只有一个）都要在图表构建阶段合并至一个操作（op）中。
            merged_summary = tf.summary.merge_all()
            #在创建好会话（session）之后，可以实例化一个tf.train.SummaryWriter，
            #用于写入包含了图表本身和即时数据具体值的事件文件。
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

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

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                #状态可视化
                #最后，每次运行summary_op时，都会往事件文件中写入最新的即时数据，
                #函数的输出会传入事件文件读写器（writer）的add_summary()函数。。
                summary_writer.add_summary(summary, step)

                #if step % 50 == 0:
                #改成每一个step输出一条，最终结果如下 996/50 = 19，打乱分10次训练，所以是19*10
                #2018-03-11T09:52:58.812304: step 190, softmax_loss 56.0816, acc 0.62
                tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                print(tempstr)
            #self.num_epochs = 10 训练十回
            for one_epoch in range(settings.num_epochs):

                #train_word为读取的narray数据
                temp_order = list(range(len(train_word)))
                #shuffle为打乱顺序函数，打乱10次
                np.random.shuffle(temp_order)
                #打乱顺序后按big_num=50一批分批次训练
                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    #因为是按big_num=50一批分批次训练，所以
                    #从      i * settings.big_num
                    #到(i + 1) * settings.big_num
                    #temp_order为打乱顺序后的输入train_x列表
                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    #将输入的nparray赋成列表
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    #统计单个单词的字符数
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    #train_step的参数都是np.array数据类型，即数组类型
                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    #为了节省不必要的空间，现在的设定是8000以后的step才会每过100个step存储一次模型。
                    if current_step > 100 and current_step % 5 == 0:

                        print('saving model')
                        path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)
                    #太早开始存储模型，早期的模型效果非常差，没有用处的；这就是为什么我设置8000以后才会存储模型。
                    #8000以后的每隔100存储的模型，可以互相比较效果取最好的，因为有可能太后期的模型又会overfit.
                    #项目中的training data仅仅是示例，远远不够训练出一个可用的模型。
                    #step>8000是在足够多的训练集上训练时用到的参数。

    endtime = datetime.datetime.now().isoformat()
    print(endtime)

    #print((endtime - starttime).seconds)


if __name__ == "__main__":
    tf.app.run()