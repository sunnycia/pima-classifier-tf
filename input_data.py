import numpy as np
import os

def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class DataSet(object):
    def __init__(self, datas, labels):
        assert datas.shape[0] == labels.shape[0], (
          "datas.shape: %s labels.shape: %s" % (datas.shape,
                                                 labels.shape))
        self._num_examples = datas.shape[0]
        self._datas = datas.astype(np.float32)
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def datas(self):
        return self._datas
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            # print "A epoch is complete. Now shuffle the set and begin next epoch"
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._datas = self._datas[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._datas[start:end], self._labels[start:end]



filePath = "Pima-training-set.txt"
def read_dataset(filePath):
    
    f = open(filePath)
    line = f.readline()
    count = 0
    labellist = []
    while line:
        linelist = line.split()
        dat_arr = np.array(linelist)[:8]
        lbl = int(linelist[8]) - 1
        # print lbl;exit()
        # print dat_arr.shape
        if not count == 0:
            data = np.concatenate((data, dat_arr[np.newaxis, ...]), axis=0)
            # print type(labellist)
            labellist.append(lbl)
        else:
            data = dat_arr[np.newaxis, ...]
            labellist = [lbl]
            
            print type(labellist),labellist;
        count += 1
        line = f.readline()
    label = dense_to_one_hot(np.array(labellist))
    # print label.shape;
    # print label;exit()
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    label = label[perm]
    # print data.shape
    # print label.shape
    return DataSet(data, label)
# read_dataset(filePath)