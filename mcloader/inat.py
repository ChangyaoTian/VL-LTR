import os
import os.path as osp
import mmcv

from .image_list import ImageList


class iNat(ImageList):

    def __init__(self, root, json_file, select=False):
        data = mmcv.load(osp.join(root, json_file))
        images = data['images']
        annotations = data['annotations']
        img_list = []
        count = {}
        for img, ann in zip(images, annotations):
            category_id = ann['category_id']
            if select:
                if category_id not in count:
                    count[category_id] = 0
                count[category_id] += 1
                if count[category_id] > 50:
                    continue
            img_list.append((img['file_name'], category_id))

        super(iNat, self).__init__(root, img_list)
