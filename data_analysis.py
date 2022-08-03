import glob
import os
import json
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from itertools import chain


# 一、将训练集的作者id和对应段落整理到一个文档，统计段落总数并检测训练集中是否缺失标签
def task_one():
    input_train = 'pan22/dataset1/train'
    output_analysis = 'data_analysis'
    corpora = glob.glob(input_train + '/*.txt')
    index = 0
    para_num_dict = {}
    para_dict = {}
    paras = {}
    for document_path in corpora:
        # 读取每一个文本并赋予对应id
        file = open(document_path, 'r', encoding="utf-8")
        document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        paragraphs = document.split('\n')
        para_num_dict[share_id] = len(paragraphs)
        paras[share_id] = paragraphs
    for i in range(1, 1401):
        for paragraph in paras[str(i)]:
            para_dict[index] = paragraph
            index += 1
    lost_label_document_id = []
    truth_label = read_ground_truth_files(input_train)
    author_list = []
    for num in range(1, 1401):
        author_ids = truth_label[str(num)]["structure"]
        author_local_change = truth_label[str(num)]['changes']
        if len(author_local_change)+1 != para_num_dict[str(num)]:
            lost_label_document_id.append(num)
            author_local_change.append(0)
        author_list.append(author_ids.pop(0))
        for author_change in author_local_change:
            if author_change == 0:
                author_list.append(author_list[-1])
            else:
                author_list.append(author_ids.pop(0))

    result = "number of paragraphs: {}\nnumber of author label: {}\ndocument id of lost label: {}". \
        format(len(para_dict.keys()), len(author_list), lost_label_document_id)
    print(result)
    with open(os.path.join(output_analysis, "new_train.txt"), 'w', encoding='utf-8') as f:
        for k, v in para_dict.items():
            f.write('%d\t%s\n' % (author_list[k], v))
    f.close()


# 二、统计训练集每篇文本的段落数，每个段落的句子数，每条句子的长度，对应的平均值、中值、众数
def task_two():
    input_train = 'pan22/dataset1/validation'
    output_train = 'data_analysis'
    corpora = glob.glob(input_train + '/*.txt')

    para_num_dict = {}
    for document_path in corpora:
        # 读取每一个文本并赋予对应id
        with open(document_path, 'r', encoding="utf-8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        paragraphs = document.split('\n')
        para_num_dict[int(share_id)] = len(paragraphs)
        # for paragraph in paragraphs:
        #     sentences = split_into_sentences(paragraph)
        #     sen_num_list.append(len(sentences))
        #     for sentence in sentences:
        #         sentence = sentence.split(' ')
        #         sen_len_list.append(len(sentence))

    # para_info = get_info(para_num_list)
    # sen_num_info = get_info(sen_num_list)
    # sen_len_info = get_info(sen_len_list)
    # result = 'paragraphs information:\n{}\nnumber of sentences for each paragraph:\n{}\nsentence length:\n{}'.\
    #     format(para_info, sen_num_info, sen_len_info)
    # with open(os.path.join(output_train, "analysis2.txt"), 'w') as f:
    #     f.write(result)
    # f.close()
    print(sum(para_num_dict.values()))
    # 画图
    # author_changes = sorted(para_num_dict.items(), key=lambda e: e[0])
    # x = []
    # y = []
    # for i in range(max(para_num_dict.keys())):
    #     x.append(author_changes[i][0])
    #     y.append(author_changes[i][1])
    # plt.title("The number of paragraphs per document")
    # plt.xlabel("Document id")
    # plt.ylabel("The number of paragraphs")
    # plt.plot(x, y, 'ro')
    # x_major_locator = MultipleLocator(2000)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.show()


# 三、统计一篇文章中有几位作者，画出全部文章作者数量分布图，风格变化分布图
def task_three():
    input_train = 'pan22/dataset1/train'
    output_train = 'data_analysis'
    truth_label = read_ground_truth_files(input_train)
    author_num_dict = {}
    author_changes_dict = {}
    for num in range(1401):
        if num == 0:
            continue
        author_changes = truth_label[str(num)]['changes']
        author_num = truth_label[str(num)]["authors"]
        change_num = 0
        for change in author_changes:
            if change == 1:
                change_num += 1
        if change_num not in author_changes_dict.keys():
            author_changes_dict[change_num] = 1
        else:
            author_changes_dict[change_num] += 1
        if author_num not in author_num_dict.keys():
            author_num_dict[author_num] = 1
        else:
            author_num_dict[author_num] += 1
    result = 'author number distribution: \n{}\nauthor changes distribution: \n{}'.\
        format(author_num_dict, author_changes_dict)
    with open(os.path.join(output_train, "analysis3.txt"), 'w') as f3:
        f3.write(result)
    f3.close()

    # 画图
    author_changes = sorted(author_changes_dict.items(), key=lambda e: e[0])
    x = []
    y = []
    for i in range(max(author_changes_dict.keys())):
        x.append(author_changes[i][0])
        y.append(author_changes[i][1] / 11200)
    plt.title("Author changes distribution")
    plt.xlabel("Number of style changes")
    plt.ylabel("Percentage of documents")
    plt.plot(x, y, marker='o', color='r')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


# 四、检测训练集和验证集作者id是否有相同的
def task_four():
    input_train = 'pan22/dataset1/train'
    input_validation = 'pan22/dataset1/validation'
    output_analysis = 'data_analysis'
    truth_label = read_ground_truth_files(input_train)
    validation_label = read_ground_truth_files(input_validation)
    author_list_val = []
    author_list = []
    for num in range(1, 11201):
        author_ids = truth_label[str(num)]["structure"]
        for author_id in author_ids:
            if author_id not in author_list:
                author_list.append(author_id)
    for num in range(1, 2401):
        author_ids = validation_label[str(num)]["structure"]
        for author_id in author_ids:
            if author_id not in author_list:
                author_list_val.append(author_id)
    print(author_list_val)
    result = "common authors ids:{}\nnumber of author in train: {}\nnumber of author in validation: {}". \
        format(sorted(set(author_list) & set(author_list_val)), len(author_list), len(author_list_val))
    with open(os.path.join(output_analysis, "analysis4.txt"), 'w') as f:
        f.write(result)
    f.close()
    print(result)


# 五、统计训练集中有多少位作者，对应作者有多少段落
def task_five():
    output_train = 'data_analysis'
    author_dict = {}
    newtrain = open('data_analysis/new_train.txt', 'r', encoding='utf-8')
    for line in newtrain.readlines():
        label, _ = line.strip().split('\t')
        label = int(label)
        if label not in author_dict.keys():
            author_dict[label] = 1
        else:
            author_dict[label] += 1

    # result = "number of authors:{}\nnumber of author's paragraphs:{}\nspecific author paragraphs:{}".\
    #     format(len(set(author_dict.keys())), get_info(list(author_dict.values())), sorted(author_dict.items()))
    # with open(os.path.join(output_train, "analysis1.txt"), 'w') as f:
    #     f.write(result)
    # f.close()

    # 画图
    author_changes = sorted(author_dict.items(), key=lambda e: e[1])
    x = []
    y = []
    for i in range(len(author_dict.keys())):
        x.append(i+1)
        y.append(author_changes[i][1])
    plt.title("The distribution of the number of paragraphs per author")
    plt.xlabel("Author id")
    plt.ylabel("The number of paragraphs")
    plt.plot(x, y, marker='o', color='r')
    x_major_locator = MultipleLocator(2500)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


# 六、将总训练集里一些作者的段落取出来作为验证集
# 验证集大小应该为11588 = 77252*0.15，这里取大小为14552，作者个数12617(至少两个段落的作者) + 1935(只有一个段落的作者)
# 总训练集大小为77252，作者总数为17051
# 要求验证集中的这14552个作者各自的段落数小于该作者对应的总段落数，所以干脆每个作者只取一个段落作为验证集
def task_six():
    one_author_id = []
    author_dict = {}
    newtrain = open('data_analysis/new_train.txt', 'r', encoding='utf-8')
    for line in newtrain.readlines():
        la, _ = line.strip().split('\t')
        la = int(la)
        if la not in author_dict.keys():
            author_dict[la] = 1
        else:
            author_dict[la] += 1
    for k, v in author_dict.items():
        if v == 1:
            one_author_id.append(k)
    print(one_author_id)
    print(len(one_author_id))
    split_train = open('data_analysis/split_train.txt', 'a', encoding='utf-8')
    split_validation = open('data_analysis/split_validation.txt', 'a', encoding='utf-8')
    author_add = []
    count_one_author_num = 0
    split_val_num = 0
    split_train_num = 0
    with open('data_analysis/new_train.txt', 'r', encoding='utf-8') as fb:
        for line in fb.readlines():
            line_t = line.strip().split('\t')
            if int(line_t[0]) not in author_add and int(line_t[0]) not in one_author_id:
                split_validation.write(line)
                author_add.append(int(line_t[0]))
                split_val_num += 1
            elif int(line_t[0]) in one_author_id and count_one_author_num < 1935:
                count_one_author_num += 1
                split_validation.write(line)
                split_val_num += 1
                split_train_num += 1
                split_train.write(line)
            else:
                split_train_num += 1
                split_train.write(line)
    print('number of split train: %d\nnumber of split validation: %d' % (split_train_num, split_val_num))

    split_validation.close()

    split_val = open('data_analysis/split_validation.txt', 'r', encoding='utf-8')
    val_list = []
    val_ = []
    for row in split_val.readlines():
        author_id, _ = row.strip().split('\t')
        if author_id not in val_list:
            val_list.append(author_id)
        else:
            val_.append(author_id)
    print(len(val_))
    print(val_)


# 七、为了检测nan值的原因，将训练集切分成不同大小进行检测
def task_seven():
    split_train = open('data_analysis/split_train.txt', 'r', encoding='utf-8')
    split_small = open('data_analysis/split_small.txt', 'a', encoding='utf-8')
    train_list = split_train.readlines()
    for i in range(100):
        split_small.write(train_list.pop(0))


# 八、检测验证集的作者个数
def task_eight():
    split_val = open('data_analysis/split_validation.txt', 'r', encoding='utf-8')
    val_list = []
    val_ = []
    for row in split_val.readlines():
        author_id, _ = row.strip().split('\t')
        if author_id not in val_list:
            val_list.append(author_id)
        else:
            val_.append(author_id)
    print(len(val_))
    print(val_)
    print(len(set(val_list)))


# 九、给作者id添加分类序号
def task_nine():
    author_dict = {}
    newtrain = open('data_analysis/new_train.txt', 'r', encoding='utf-8')
    for line in newtrain.readlines():
        la, _ = line.strip().split('\t')
        la = int(la)
        if la not in author_dict.keys():
            author_dict[la] = 1
        else:
            author_dict[la] += 1
    print(get_info(list(author_dict.values())))


# 十、统计训练集中每一个段落的长度和对应数量，并画出分布图
def task_ten():
    # output_train = 'data_analysis'
    corpora = glob.glob('train/*.txt')
    # corpora = glob.glob('../pan21/validation/*.txt')
    para_dict = {}
    for document_path in corpora:
        with open(document_path, 'r', encoding="utf-8") as file:
            document = file.read()
        paragraphs = document.split('\n')
        for i in range(len(paragraphs)):
            para_len = len(paragraphs[i])
            if para_len > 4000:
                continue
            if para_len not in para_dict.keys():
                para_dict[para_len] = 1
            else:
                para_dict[para_len] += 1

    # 画图
    para_dict = sorted(para_dict.items(), key=lambda _: _[0])
    x = []
    y = []
    for i in range(len(para_dict)):
        x.append(para_dict[i][0])
        y.append(para_dict[i][1])
    plt.title("Distribution of paragraphs length in training set")
    plt.xlabel("Length of paragraphs")
    plt.ylabel("Number")
    plt.plot(x, y, 'or')
    x_major_locator = MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    print(get_info(y))


# 十一、统计验证集段落数与标签数是否一致
def eleven():
    corpora = glob.glob('validation/*.txt')
    train_labels = read_ground_truth_files('validation')
    para_len_list = []
    change_label_list = []
    for document_path in corpora:
        with open(document_path, 'r', encoding="utf-8") as file:
            document = file.read()
        paragraphs = document.split('\n')
        para_len = len(paragraphs)
        share_id = os.path.basename(document_path)[8:-4]
        change_labels = train_labels[share_id]['changes']
        change_label_list.append(change_labels)
        para_len_list.append(para_len)
    print(len(change_label_list))
    print(len(list(chain.from_iterable(change_label_list))))
    print(len(para_len_list))
    print(sum(para_len_list))


# 十二、统计一下训练集changes标签中1和0的个数
def twelve():
    train_labels = read_ground_truth_files('validation')
    label_dict = {'1': 0, '0': 0, 'length': 0}
    task3_label_dict = {'1': 0, '0': 0, 'length': 0}
    for i in range(1, 2401):
        labels = train_labels[str(i)]['changes']
        task3labels = train_labels[str(i)]['paragraph-authors']
        task3_labels = separate_para_label(task3labels)

        label_dict['1'] += sum(labels)
        label_dict['0'] += (len(labels) - sum(labels))
        label_dict['length'] += len(labels)

        task3_label_dict['1'] += sum(task3_labels)
        task3_label_dict['0'] += (len(task3_labels) - sum(task3_labels))
        task3_label_dict['length'] += len(task3_labels)
    print(label_dict)
    print(task3_label_dict)

    # 画图
    # para_dict = sorted(author_num_dict.items(), key=lambda _: _[0])

    x = ['        changes label (14,095)', '          task3-binary label (60,365)']
    y1 = [6550, 27727]
    y2 = [7545, 32638]
    # for i in range(2):
    #     y1.append(label_dict[str(i)])
    #     y2.append(task3_label_dict[str(i)])

    plt.title('The number of 1 and 0 in validation set')
    plt.xlabel("Label")
    plt.ylabel("Number")
    # plt.yticks([500, 1000, 1500, 2000, 2500, 2800])

    width = 0.3  # 柱子的宽度
    index = np.arange(2)
    plt.bar(index, y1, width, color='steelblue', tick_label=x)
    plt.bar(index + width, y2, width, color='red')
    plt.legend(['0', '1'])

    for a, b in zip(index, y1):  # 柱子上的数字显示
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)
    for a, b in zip(index + width, y2):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)

    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


# 十三、画出训练集和验证集的作者数量分布图
def thirteen():
    input_train = 'validation'
    truth_label = read_ground_truth_files(input_train)
    author_num_dict = {}
    for num in range(1, 2401):
        author_num = truth_label[str(num)]["authors"]
        if author_num not in author_num_dict.keys():
            author_num_dict[author_num] = 1
        else:
            author_num_dict[author_num] += 1
    result = 'author number distribution: \n{}'.\
        format(author_num_dict)
    print(result)

    # 画图
    para_dict = sorted(author_num_dict.items(), key=lambda _: _[0])
    x = []
    y = []
    for i in range(len(para_dict)):
        x.append(para_dict[i][0])
        y.append(para_dict[i][1])

    plt.bar(x, y)
    plt.title('Distribution of author number of each document in train set')
    plt.xlabel("Author number of each document")
    plt.ylabel("Number")
    # plt.yticks([500, 1000, 1500, 2000, 2500, 2800])
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


# 将task3标签拆分为task3-binary标签
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


# 把真实标签读取出来并赋予对应id，存为字典
def read_ground_truth_files(truth_folder):
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem-*.json')):
        with open(truth_file, 'r', encoding='utf-8') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[14:-5]] = curr_truth
    return truth


# 中值
def get_median(data):
    data = sorted(data)
    size = len(data)
    median = 0
    if size % 2 == 0:
        # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
    if size % 2 == 1:
        # 判断列表长度为奇数
        median = data[(size - 1) // 2]
    return median


# 众数(返回多个众数的平均值)
def get_most(list_):
    most = []
    item_num = dict((item, list_.count(item)) for item in list_)
    for k, v in item_num.items():
        if v == max(item_num.values()):
            most.append(k)
    mos_num = [sum(most) / len(most), max(item_num.values())]
    return mos_num


# 获取平均数
def get_average(list_):
    sum_ = 0
    for item in list_:
        sum_ += item
    return sum_ / len(list_)


# 整合相关信息
def get_info(x_list):
    max_len = max(x_list)
    min_len = min(x_list)
    avg_len = get_average(x_list)
    mos_num = get_most(x_list)
    med_len = get_median(x_list)
    print_format = '{{\n  max: {}\n  min: {}\n  avg: {}\n  most: {}\n  median: {}\n}}'. \
        format(max_len, min_len, avg_len, mos_num, med_len)
    return print_format


if __name__ == '__main__':
    task_one()
    # task_two()
    # task_three()
    # task_four()
    # task_five()
    # task_six()
    # task_seven()
    # task_eight()
    # task_nine()
    # task_ten()
    # eleven()
    # twelve()
    # thirteen()