import tensorflow as tf
import codecs
import sys
from configparser import ConfigParser



gConfig={}
def get_config(config_file='seq2seq.ini'):
    '''
    读取配置文件
    :param config_file: 配置文件地址
    :return:
    '''
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_file,encoding='utf8')
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)

# 定义NMTModel类来描述模型。
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
                "weight", [gConfig['HIDDEN_SIZE'], gConfig['TRG_VOCAB_SIZE']])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [gConfig['TRG_VOCAB_SIZE']])

    def inference(self, src_input):
        # 将输入句子整理为大小为1的batch。
        src_input = tf.expand_dims(src_input,0)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时相同。
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, dtype=tf.float32) #src_size

        # 设置解码的最大步数。
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, gConfig['SOS_ID'])

            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), gConfig['EOS_ID']),
                    tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                dec_outputs, next_state = self.dec_cell.call(
                    state=state, inputs=trg_emb)
                output = tf.reshape(dec_outputs, [-1, gConfig['HIDDEN_SIZE']])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def main():
    global gConfig
    gConfig = get_config('seq2seq.ini')

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

    # 建立解码所需的计算图。
    enids = tf.placeholder(tf.int32,[None])
    output_op = model.inference(enids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, gConfig['CHECKPOINT_PATH'])

    # 定义个测试句子。
    test_en_text = "This is a test . <eos>"
    print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(gConfig['SRC_VOCAB'], "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]
    print(test_en_ids)

    # 读取翻译结果。
    output_ids = sess.run(output_op,feed_dict={enids:test_en_ids})
    print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字。
    with codecs.open(gConfig['TRG_VOCAB'], "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])

    # 输出翻译结果。
    print(output_text.encode('utf8').decode(sys.stdout.encoding))

    txt_input="This is a test ."
    while(txt_input!='00000'):
        txt_input=input('请输入待翻译英文句子')+' <eos>'
        test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                       for token in txt_input.split()]
        output_ids = sess.run(output_op, feed_dict={enids: test_en_ids})
        output_text = ''.join([trg_vocab[x] for x in output_ids])
        print(output_text.encode('utf8').decode(sys.stdout.encoding))

    sess.close()


if __name__ == "__main__":
    main()
