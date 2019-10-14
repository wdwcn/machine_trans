import tensorflow as tf
from configparser import ConfigParser

gConfig={}
def get_config(config_file='seq2seq.ini'):
    '''
    读取配置文件
    :param config_file: 配置文件地址
    :return:
    '''
    parser = ConfigParser()
    #保留参数的大小写
    parser.optionxform = str
    parser.read(config_file,encoding='utf8')
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)


# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
# batching操作。
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
    # 由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空（只包含<EOS>）的句子和长度过长的句子。
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len,gConfig['MAX_LEN']))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len,gConfig['MAX_LEN']))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FilterLength)

    # 从图9-5可知，解码器需要两种格式的目标句子：
    #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
    #   2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
    # 形式并加入到Dataset中。
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[gConfig['SOS_ID']], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset

class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(gConfig['HIDDEN_SIZE'])
             for _ in range(gConfig['NUM_LAYERS'])])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(gConfig['HIDDEN_SIZE'])
             for _ in range(gConfig['NUM_LAYERS'])])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [gConfig['SRC_VOCAB_SIZE'], gConfig['HIDDEN_SIZE']])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [gConfig['TRG_VOCAB_SIZE'], gConfig['HIDDEN_SIZE']])

        # 定义softmax层的变量
        if gConfig['SHARE_EMB_AND_SOFTMAX']:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight",[gConfig['HIDDEN_SIZE'], gConfig['TRG_VOCAB_SIZE']])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [gConfig['TRG_VOCAB_SIZE']])

    # 在forward函数中定义模型的前向计算图。
    # src_input, src_size, trg_input, trg_label, trg_size分别是上面
    # MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout。
        src_emb = tf.nn.dropout(src_emb, gConfig['KEEP_PROB'])
        trg_emb = tf.nn.dropout(trg_emb, gConfig['KEEP_PROB'])


        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        # 计算解码器每一步的log perplexity。这一步与语言模型代码相同。
        output = tf.reshape(dec_outputs, [-1, gConfig['HIDDEN_SIZE']])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)


        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, gConfig['MAX_GRAD_NORM'])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))
        return cost_per_token, train_op



def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch。
    # 重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session,gConfig['MODEL_PATH'], global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    #初始化参数
    global gConfig
    gConfig = get_config('seq2seq.ini')
    print(gConfig.keys())

    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None,
                           initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据。
    data = MakeSrcTrgDataset(gConfig['SRC_TRAIN_DATA'],gConfig['TRG_TRAIN_DATA'], gConfig['BATCH_SIZE'])
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(gConfig['NUM_EPOCH']):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == "__main__":
    main()
