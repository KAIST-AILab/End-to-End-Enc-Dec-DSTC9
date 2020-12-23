from itertools import count, repeat
import json
import os


class DatasetWalker(object):
    def __init__(self, dataset, dataroot, start=0.0, end=1.0, labels=False, labels_file=None, tfidf=False,
                 tfidf_file=None):
        path = os.path.join(os.path.abspath(dataroot))

        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % dataset)

        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.start = round(len(self.logs) * start)
        self.end = round(len(self.logs) * end)
        self.logs = self.logs[self.start:self.end]

        self.labels = None
        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
            self.labels = self.labels[self.start:self.end]

        self.tfidf = None
        if tfidf is True:
            if tfidf_file is None:
                tfidf_file = os.path.join(path, dataset, 'tfidf.json')

            with open(tfidf_file, 'r') as f:
                self.tfidf = json.load(f)
            self.tfidf = self.tfidf[self.start:self.end]

    def __iter__(self):
        names = ['id', 'log', 'label']
        iterators = [count(), self.logs, self.labels or repeat(None)]
        if self.tfidf:
            names.append('tfidf')
            iterators.append(self.tfidf)
        yield from (dict(zip(names, item)) for item in zip(*iterators))

    def __len__(self, ):
        return len(self.logs)
