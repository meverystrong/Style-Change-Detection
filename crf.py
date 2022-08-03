import glob
import os
import json

from bert4keras.backend import keras, set_gelu, K
from bert4keras.layers import ConditionalRandomField
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense, Lambda
from ner.client import NER
from tqdm import tqdm

set_gelu('tanh')  # 切换gelu版本


maxlen = 64
batch_size = 32
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
num_classes = 2
epochs = 2
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
categories = set()


config_path = 'S:/python/model/electra_base/electra_config.json'
checkpoint_path = 'S:/python/model/electra_base/electra_base'
dict_path = 'S:/python/model/electra_base/vocab.txt'


train_path = 'pan22/dataset3/train'
validation_path = 'pan22/dataset3/validation'
test_path = 'pan22/dataset3/test'


def read_label(label_file):
    labels = {}
    for label in glob.glob(os.path.join(label_file, 'truth-problem-*.json')):
        with open(label, 'r', encoding='utf-8') as lf:
            curr_label = json.load(lf)
            labels[os.path.basename(label)[14:-5]] = curr_label
    return labels


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


def train_load(filename):
    train_labels = read_label(filename)
    data_plus = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        para_list = document.split('\n')
        author_labels = train_labels[share_id]['paragraph-authors']
        separate_labels = separate_para_label(author_labels)
        for id, para in enumerate(para_list):
            for i in range(id):
                data_plus.append((para_list[i], para_list[id], int(separate_labels.pop(0))))
    return data_plus


def val_load(filename):
    train_labels = read_label(filename)
    data = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        para_list = document.split('\n')
        change_labels = train_labels[share_id]['changes']
        para_pre = None
        for id, para in enumerate(para_list):
            if id == 0:
                para_pre = para
                continue
            label = change_labels[id-1]
            para_curr = para
            data.append((para_pre, para_curr, int(label)))
            para_pre = para
    return data


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


train_data = train_load(train_path)
valid_data = val_load(validation_path)
categories = list(sorted(categories))


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(valid_data, batch_size)


# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True
)


# 建立模型，加载权重
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False
)

output = Lambda(
    lambda x: x[:, 0],
    name='CLS-token'
)(model.model.output)

output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=model.initializer
)(output)

CRF = ConditionalRandomField(
    lr_multiplier=crf_lr_multiplier
)

output = CRF(output)

model = keras.models.Model(model.model.input, output)
model.summary()

AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    metrics=['accuracy'],
)


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('finetune_change3/best_model.weights')
            model.save('finetune_change3/best_model.h5')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_generator)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('finetune_change3/best_model.weights')
    with open('finetune_change3/result.txt', 'a', encoding='utf-8') as result_file:
        result_file.write(u'final test acc: %05f\n' % (evaluate(test_generator)))
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))