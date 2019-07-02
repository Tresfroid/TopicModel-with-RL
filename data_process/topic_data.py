import os
import re
import codecs
import copy
from collections import Counter
import numpy as np
from scipy import sparse
from spacy.lang.en import English
import config.file_handing as fh


def load_and_process_data(infile, vocab_size, parser, output_dir, vocab=None, log_transform=None, label_list=None):
    mallet_stopwords = fh.read_text('/data/lengjia/topic_model/mallet_stopwords.txt')  # 得到文件每一行组成的list
    mallet_stopwords = {s.strip() for s in mallet_stopwords}

    data = fh.read_json(infile)
    n_items = len(data)
    print("Parsing %d documents" % n_items)

    # vocab = fh.read_json(os.path.join(output_dir, 'train.vocab.json'))

    parsed = []  ##存储处理过的词
    labels = []
    word_counts = Counter()
    doc_counts = Counter()
    keys = list(data.keys())
    keys.sort()

    # vocab_dict = {}
    for i, k in enumerate(keys):
        item = data[k]
        if i % 1000 == 0 and i > 0:
            print(i)

        text = item['text']
        label = item['label']
        labels.append(label)

        # remove each pair of angle brackets and everything within them
        text = re.sub('<[^>]+>', '', text)
        parse = parser(text)
        # remove white space from tokens
        words = [re.sub('\s', '', token.orth_) for token in parse]
        words = [word.lower() for word in words if len(word) >= 1]
        words = [word for word in words if len(word) <= 20]
        words = [word for word in words if word not in mallet_stopwords]
        # remove tokens that don't contain letters or numbers
        #     words = [word for word in words if re.match('^[a-zA-A]*$', word) is not None]
        words = [word for word in words if re.match('[a-zA-A0-9]', word) is not None]
        # convert numbers to a number symbol
        words = [word for word in words if re.match('[0-9]', word) is None]
        words = [word for word in words if re.search('@', word) is None]  ##delete string with @
        # store the parsed documents
        parsed.append(words)
        # keep track fo the number of documents with each word
        word_counts.update(words)
        doc_counts.update(set(words))

    print("Size of full vocabulary=%d" % len(word_counts))

    if vocab is None:
        initial_vocab = {}
        vocab = copy.copy(initial_vocab)
        for w in word_counts.most_common(vocab_size):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocab_size:
                    break
        total_words = np.sum(list(vocab.values()))
        word_freqs = np.array([vocab[v] for v in vocab.keys()]) / float(total_words)  # 词频
    else:
        word_freqs = None

    if label_list is None:
        label_list = list(set(labels))
        label_list.sort()
    n_labels = len(label_list)
    label_index = dict(zip(label_list, range(n_labels)))

    X = np.zeros([n_items, vocab_size], dtype=int)  # 数据条数 * 词数    值表示出现的次数
    y = []  # label  数据条数 * label数20  值为出现的次数
    lists_of_indices = []  # an alternative representation of each document as a list of indices

    counter = Counter()
    print("Converting to count representations")
    count = 0
    total_tokens = 0
    for i, words in enumerate(parsed):
        indices = [vocab[word] for word in words if word in vocab]  ##存放word的index
        word_subset = [word for word in words if word in vocab]  # 存放word
        counter.clear()
        counter.update(indices)  ##word的index：出现次数
        if len(counter.keys()) > 0:
            values = list(counter.values())
            if log_transform:
                values = np.array(np.round(np.log(1 + np.array(values, dtype='float'))), dtype=int)
            X[np.ones(len(counter.keys()), dtype=int) * count, list(counter.keys())] += values
            total_tokens += len(word_subset)
            y_vector = np.zeros(n_labels)
            y_vector[label_index[labels[i]]] = 1
            y.append(y_vector)
            lists_of_indices.append(indices)
            count += 1

    print("Found %d non-empty documents" % count)
    print("Total tokens = %d" % total_tokens)

    # drop the items that don't have any words in the vocabualry
    X = np.array(X[:count, :], dtype=int)
    X_indices = X.copy()
    X_indices[X_indices > 0] = 1
    print(X.shape)
    temp = np.array(y)
    y = np.array(temp[:count], dtype=int)
    print(y.shape)
    sparse_y = sparse.csr_matrix(y)
    sparse_X = sparse.csr_matrix(X)
    sparse_X_indices = sparse.csr_matrix(X_indices)

    return sparse_X, vocab, lists_of_indices, sparse_y, word_freqs, label_list, sparse_X_indices


def preprocess_data(train_infile, test_infile, output_dir, vocab_size, log_transform=False):
    print("Loading Spacy")
    parser = English()

    with codecs.open(train_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    train_n_items = len(lines)
    train_indices = list(set(range(train_n_items)))

    with codecs.open(test_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    test_n_items = len(lines)
    test_indices = list(set(range(test_n_items)))  # [0,1,...,n_items-1]

    train_X, train_vocab, train_indices, train_y, word_freqs, label_list, train_X_indices = load_and_process_data(
        train_infile, vocab_size, parser, output_dir, log_transform=log_transform)
    test_X, _, test_indices, test_y, _, _, test_X_indices = load_and_process_data(test_infile, vocab_size, parser,
                                                                                  output_dir, vocab=train_vocab,
                                                                                  log_transform=log_transform,
                                                                                  label_list=label_list)

    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'))
    fh.write_to_json(train_indices, os.path.join(output_dir, 'train.indices.json'))
    fh.save_sparse(train_y, os.path.join(output_dir, 'train.labels.npz'))
    fh.save_sparse(train_X_indices, os.path.join(output_dir, 'train_X_indices.npz'))

    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_indices, os.path.join(output_dir, 'test.indices.json'))
    fh.save_sparse(test_y, os.path.join(output_dir, 'test.labels.npz'))
    fh.save_sparse(test_X_indices, os.path.join(output_dir, 'test_X_indices.npz'))

    #     save_sparse(dev_X, os.path.join(output_dir, 'dev.npz'))
    #     write_to_json(dev_indices, os.path.join(output_dir, 'dev.indices.json'))
    #     save_sparse(dev_y, os.path.join(output_dir, 'dev.labels.npz'))
    #     save_sparse(dev_X_indices, os.path.join(output_dir, 'dev_X_indices.npz'))

    n_labels = len(label_list)
    label_dict = dict(zip(range(n_labels), label_list))
    fh.write_to_json(label_dict, os.path.join(output_dir, 'train.label_list.json'))
    # fh.write_to_json(list(word_freqs.tolist()), os.path.join(output_dir, 'train.word_freq.json'))
