import os
import os.path as osp
import pickle

from torch.utils.data import Dataset

from .imagenet import ImageNet
from .image_list import ImageList
from .inat import iNat
from .preprocess import SentPreProcessor


def get_sentence_tokens(dataset: str, desc_path, context_length):
    print('using clip text tokens splitted by sentence')
    cache_root = 'cached'
    cache_path = osp.join(cache_root, '%s_desc_text_sent.pkl' % dataset)
    clip_token_path = osp.join(cache_root, '%s_text_tokens.pkl' % dataset)
    if osp.exists(clip_token_path):
        with open(clip_token_path, 'rb') as f:
            text_tokens = pickle.load(f)
        return text_tokens

    preprocessor = SentPreProcessor(root=desc_path, dataset=dataset)
    if not osp.exists(cache_path):
        os.makedirs(cache_root, exist_ok=True)
        texts = preprocessor.get_clip_text()
        texts = preprocessor.split_sent(texts)
        with open(cache_path, 'wb') as f:
            pickle.dump(texts, f)
    else:
        with open(cache_path, 'rb') as f:
            texts = pickle.load(f)
    text_tokens = preprocessor.tokenize(texts, context_length=context_length)
    with open(clip_token_path, 'wb') as f:
        pickle.dump(text_tokens, f)
    return text_tokens


class ClassificationDataset(Dataset):
    """Dataset for classification.
    """

    def __init__(self, dataset='IMNET', split='train', nb_classes=1000,
                 desc_path='', context_length=0, pipeline=None, select=False):
        assert dataset in ['PLACES_LT', "IMNET", "IMNET_LT", "INAT"]
        self.nb_classes = nb_classes
        if dataset == 'IMNET':
            self.data_source = ImageNet(root='data/imagenet/%s' % split,
                                        list_file='data/imagenet/meta/%s.txt' % split,
                                        select=select)
        elif dataset == 'IMNET_LT':
            self.data_source = ImageNet(root='data/imagenet',
                                        list_file='data/imagenet/ImageNet_LT_%s.txt' % split)
        elif dataset == 'INAT':
            self.data_source = iNat(root='data/iNat/',
                                    json_file='%s2018.json' % split,
                                    select=select)
        elif dataset == 'PLACES_LT':
            self.data_source = ImageList(root='data/places',
                                        list_file='data/places/Places_LT_%s.txt' % split,
                                        select=select)
        
        self.text_tokens = get_sentence_tokens(dataset, desc_path, context_length)
        self.end_idxs = [len(sents) for sents in self.text_tokens]

        self.pipeline = pipeline
        assert self.data_source.labels is not None
        self.targets = self.data_source.labels

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)

        return img, target
