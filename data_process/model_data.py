import config.file_handing as fh
import numpy as np
import os
import gc

def load_data(input_dir, input_prefix, log_file, vocab=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    X = np.array(temp, dtype='float32')  # 矩阵：数据条数 * 词数    值表示出现的次数
    del temp
    temp2 = fh.load_sparse(os.path.join(input_dir, input_prefix + '_X_indices.npz')).todense()
    indices = np.array(temp2, dtype='float32')
    del temp2
    lists_of_indices = fh.read_json(os.path.join(input_dir, input_prefix + '.indices.json'))
    index_arrays = [np.array(l, dtype='int32') for l in lists_of_indices]
    del lists_of_indices
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))  # 排序后词表
    n_items, vocab_size = X.shape
    #     assert vocab_size == len(vocab)

    label_file = os.path.join(input_dir, input_prefix + '.labels.npz')
    if os.path.exists(label_file):
        print("Loading labels")
        temp = fh.load_sparse(label_file).todense()
        labels = np.array(temp, dtype='float32')  ##数据条数 * label数20
    else:
        print("Label file not found")
        labels = np.zeros([n_items, 1], dtype='float32')
    assert len(labels) == n_items
    gc.collect()

    return X, vocab, labels, indices, index_arrays

class DataIter(object):
    def __init__(self, data, indices, index_arrays, batch_size):
        self.data = data
        self.indices = indices
        self.index_arrays = index_arrays
        assert len(self.data) == len(self.indices)
        assert len(self.data) == len(self.index_arrays)
        self.batch_size = batch_size
        self.num_document = len(self.data)
        self.num_batch = self.num_document // self.batch_size
    def __iter__(self):
        self.batch_permuted_list = np.random.permutation(self.num_batch)
        self.batch_i = 0
        return self
    def __next__(self):
        if self.batch_i >= self.num_batch:
            raise StopIteration
        else:
            self.batch_index = self.batch_permuted_list[self.batch_i]
            starting_point = self.batch_index * self.batch_size
            if starting_point + self.batch_size >= self.num_document:
                end_point = self.num_document
            else:
                end_point = starting_point + self.batch_size
            batch_data = self.data[starting_point:end_point]
            batch_indices = self.indices[starting_point:end_point]
            batch_index_arrays = self.index_arrays[starting_point:end_point]
            self.batch_i += 1
        return batch_data,batch_indices,batch_index_arrays