import codecs
import collections
from operator import itemgetter


def create_vocab(raw_data,vocab_path,vocab_size):
    '''
    生成词典
    :param raw_data:
    :param vocab_path:
    :param vocab_size:
    :return:
    '''
    counter = collections.Counter()
    with codecs.open(raw_data, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序。
    sorted_word_to_cnt = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    #插入特殊字符
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > vocab_size:
        sorted_words = sorted_words[:vocab_size]

    #保存词汇表
    with codecs.open(vocab_path, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

def data_trans_by_vocab(raw_data,vocab_path,trans_data_path):
    """
    生成用于训练的数据
    :param raw_data:
    :param vocab_path:
    :param trans_data_path:
    :return:
    """
    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(vocab_path, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    #数据替换
    with codecs.open(raw_data, "r", "utf-8") as fin:
        with codecs.open(trans_data_path, 'w', 'utf-8') as fout:
            for line in fin:
                words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
                # 将每个单词替换为词汇表中的编号
                out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
                fout.write(out_line)




if __name__=='__main__':

    #生成字典
    raw_data_zh = "./data_corpus/train.txt.zh"
    vocab_output_zh = "./data_corpus/zh.vocab"
    vocab_size_zh = 4000
    raw_data_en = "./data_corpus/train.txt.en"
    vocab_output_en = "./data_corpus/en.vocab"
    vocab_size_en = 10000
    #中文字典、英文字典生成
    create_vocab(raw_data_zh, vocab_output_zh, vocab_size_zh)
    create_vocab(raw_data_en, vocab_output_en, vocab_size_en)

    #元数据转换
    trans_data_zh = "./data_corpus/train.zh"
    trans_data_en = "./data_corpus/train.en"

    data_trans_by_vocab(raw_data_zh, vocab_output_zh, trans_data_zh)
    data_trans_by_vocab(raw_data_en, vocab_output_en, trans_data_en)


