import os
from PIL import Image


class ImageList(object):

    def __init__(self, root, list_file, select=False):
        if isinstance(list_file, str):
            with open(list_file, 'r') as f:
                lines = f.readlines()
            self.has_labels = len(lines[0].split()) == 2
            if self.has_labels:
                self.fns, self.labels = zip(*[l.strip().split() for l in lines])
                self.labels = [int(l) for l in self.labels]
            else:
                self.fns = [l.strip() for l in lines]
        elif isinstance(list_file, list):
            self.has_labels = len(list_file[0]) == 2
            if self.has_labels:
                self.fns, self.labels = zip(*list_file)
            else:
                self.fns = list_file
        
        if select:
            assert self.has_labels
            n_fns = []
            n_labels = []
            cls_cnt_dict = {}
            for fns, label in zip(self.fns, self.labels):
                if label not in cls_cnt_dict:
                    cls_cnt_dict[label] = 0
                cls_cnt_dict[label] += 1
                if cls_cnt_dict[label] > 50: continue
                n_fns.append(fns)
                n_labels.append(label)
            self.fns = n_fns
            self.labels = n_labels
        
        self.fns = [os.path.join(root, fn) for fn in self.fns]

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img
