import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


class RunningMean:
    def __init__(self):
        self.mean = 0.0
        self.n = 0

    def __iadd__(self, value):
        self.mean = (float(value) + self.mean * self.n)/(self.n + 1)
        self.n += 1
        return self

    def reset(self):
        self.mean = 0.0
        self.n = 0

    def mean(self):
        return self.mean


class RunningMeanTorch:
    def __init__(self):
        self.values = []

    def __iadd__(self, value):
        with torch.no_grad():
            self.values.append(value.detach().cpu().unsqueeze(0))
            return self

    def reset(self):
        self.values = []

    def mean(self,dim=[]):
        with torch.no_grad():
            if len(self.values) == 0:
                return 0.0
            return torch.cat(self.values).mean(dim=dim).numpy()


class LossTracker:
    def __init__(self, output_folder='.',test=False):
        self.tracks = OrderedDict()
        self.n_iters = []
        self.means_over_n_iters = OrderedDict()
        self.output_folder = output_folder
        self.filename = 'log_test.csv' if test else 'log_train.csv'

    def update(self, d):
        for k, v in d.items():
            if k not in self.tracks:
                self.add(k)
            self.tracks[k] += v

    def add(self, name, pytorch=True):
        assert name not in self.tracks, "Name is already used"
        if pytorch:
            track = RunningMeanTorch()
        else:
            track = RunningMean()
        self.tracks[name] = track
        self.means_over_n_iters[name] = []
        return track

    def register_means(self, n_iter,suffix = 'iter'):
        print ('registering means,n_iter,self.n_iters',n_iter,self.n_iters)
        #for multi patient, same n_iters keep occuring! should modify!
        if n_iter not in self.n_iters:
            self.n_iters.append(n_iter)

        for key in self.means_over_n_iters.keys():
            if key in self.tracks:
                value = self.tracks[key]
                self.means_over_n_iters[key].append(value.mean(dim=0))
                value.reset()
            else:
                self.means_over_n_iters[key].append(None)
        with open(os.path.join(self.output_folder,  suffix+'_'+self.filename), mode='w') as csv_file:
            fieldnames = ['n_iter'] + [key+str(i) for key in list(self.tracks.keys()) for i in range(self.means_over_n_iters[key][0].size)]
            # fieldnames = ['n_iter'] + [list(self.tracks.keys())]
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for key in self.means_over_n_iters.keys():
                print (n_iter, key, self.means_over_n_iters[key][-1])
            for i in range(len(self.n_iters)):
                writer.writerow([self.n_iters[i]] + [self.means_over_n_iters[x][i][j] if \
                    self.means_over_n_iters[x][i].size>1 else self.means_over_n_iters[x][i] \
                        for x in self.tracks.keys() for j in range(self.means_over_n_iters[x][i].size) ])
    def __str__(self):
        result = ""
        for key, value in self.tracks.items():
            result += "%s: %.7f, " % (key, value.mean())
        return result[:-2]

    def plot(self):
        plt.figure(figsize=(12, 8))
        for key in self.tracks.keys():
            plt.plot(self.n_iters, self.means_over_n_iters[key], label=key)

        plt.xlabel('n_iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_folder, 'plot.png'))
        plt.close()

    def state_dict(self):
        return {
            'tracks': self.tracks,
            'n_iters': self.n_iters,
            'means_over_n_iters': self.means_over_n_iters}

    def load_state_dict(self, state_dict):
        self.tracks = state_dict['tracks']
        self.n_iters = state_dict['n_iters']
        self.means_over_n_iters = state_dict['means_over_n_iters']

        counts = list(map(len, self.means_over_n_iters.values()))

        if len(counts) == 0:
            counts = [0]
        m = min(counts)

        if m < len(self.n_iters):
            self.n_iters = self.n_iters[:m]

        for key in self.means_over_n_iters.keys():
            if len(self.means_over_n_iters[key]) > m:
                self.means_over_n_iters[key] = self.means_over_n_iters[key][:m]
