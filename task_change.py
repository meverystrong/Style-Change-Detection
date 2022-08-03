import glob
import os
import json
import argparse
import numpy as np

from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense, Lambda
from itertools import chain


set_gelu('tanh')

maxlen = 64
batch_size = 32
num_classes = 2


config_path = 'S:/python/model/electra_large/electra_config.json'
checkpoint_path = 'S:/python/model/electra_large/electra_large'
dict_path = 'S:/python/model/electra_large/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True
)


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False
)

# 建立模型，加载权重
output = Lambda(
    lambda x: x[:, 0],
    name='CLS-token'
)(model.model.output)

output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=model.initializer
)(output)

model = keras.models.Model(model.model.input, output)


def separate_para_label(paragraphs_label):
    separate_label = []
    for i in range(len(paragraphs_label)):
        if i == 0:
            continue
        for a in range(i):
            if paragraphs_label[a] != paragraphs_label[i]:
                separate_label.append(1)
            else:
                separate_label.append(0)
    return separate_label


def data_load(filename):
    data = []
    data_plus = []
    para_len_plus = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        share_id = os.path.basename(document_path)[8:-4]
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        para_list = document.split('\n')
        if para_list[-1] == '':
            para_list.pop(-1)
        author_labels = [1]*len(para_list)
        separate_labels = separate_para_label(author_labels)
        para_len_plus.append((share_id, len(para_list)-1, len(separate_labels)))
        para_pre = None
        for id, para in enumerate(para_list):
            if id == 0:
                para_pre = para
                continue
            para_curr = para
            data.append((para_pre, para_curr, 1))
            para_pre = para
            for i in range(id):
                data_plus.append((para_list[i], para_list[id], 1))
    return data, data_plus, para_len_plus


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def compound_label(separate_label_list, length):
    paragraphs_label = [1]
    each_para_label = 0
    dict_label = {}
    pre_index_left = 0
    pre_index_right = 0
    pre_index = 0
    for index in range(length):
        if index == 0:
            pre_index_left = index
            pre_index_right = 0
            pre_index = 0
            continue
        dict_label[index] = separate_label_list[(pre_index_left + pre_index): (pre_index_right + index)]
        pre_index_left = pre_index_left + index
        pre_index_right = pre_index_right + index
    for k, v in dict_label.items():
        for n in range(k):
            if v[n] == 1:
                got = 0
                for x in range(len(v[:n])):
                    if v[x] == 0:
                        got = 1
                if got == 1:
                    continue
                each_para_label = max(paragraphs_label) + 1
            else:
                each_para_label = paragraphs_label[n]
                break
        paragraphs_label.append(each_para_label)
    return paragraphs_label


def predict(data):
    y_pred_list = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_pred_list.append(list(y_pred))
    return list(chain.from_iterable(y_pred_list))


def save_result(changes_list, author_sep_list, para_len_plus, outputpath):
    for share_id, l, p in para_len_plus:
        changes_label = []
        author_label = []
        for i in range(l):
            changes_label.append(changes_list.pop(0))
        for i in range(p):
            author_label.append(author_sep_list.pop(0))
        solution = {
            'changes': changes_label
        }
        file_name = outputpath + '/solution-problem-' + share_id + '.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def predict_task(input_path, output_path):
    model.load_weights('S:/python/style_change_detection2022/finetune_task1/best_model.weights')

    test_data, test_data_plus, para_len_plus = data_load(input_path)
    test_generator = data_generator(test_data, batch_size)
    test_plus_generator = data_generator(test_data_plus, batch_size)
    predict_result = predict(test_generator)
    predict_plus_result = predict(test_plus_generator)
    save_result(predict_result, predict_plus_result, para_len_plus, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2022')
    parser.add_argument('-c', type=str,
                        help='Path to the test txt')
    parser.add_argument('-o', type=str,
                        help='Path to the dir with output')
    args = parser.parse_args()

    input_path = 'S:/python/style_change_detection2022/pan22/dataset1/test'
    output_path = 'S:/python/style_change_detection2022/output22/task1_large_64+32'
    predict_task(input_path, output_path)
