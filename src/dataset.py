import torch
from torchvision.datasets import Kitti
from torch.utils.data.dataloader import default_collate
import os

def collate_fn(batch):
    """
    image, image_id, (height, width), boxes, labels
    """
    items = list(zip(*batch))    
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)]) # tensor
    
    items[1] = list([i for i in items[1] if i]) # list
    
    items[2] = list([i for i in items[2] if i]) # list
    
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)]) # tensor
    
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)]) # tensor

    return items


class KittiDataset(Kitti):
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    
    def __init__(self,root, train = True, transform= None):
        super(KittiDataset, self).__init__(root)
        
        self.images = []
        self.targets = []
        self.train = train
        self._location = "training" if self.train else "testing"
        self._load_categories()
        self.transform = transform
        self.root = root
        
        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
            
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(
                    os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt")
                )

        
    @property            
    def _raw_folder(self):
        return os.path.join(self.root, "Kitti", "raw")

        

    def _load_categories(self):

        categories = ['Car','DontCare','Person_sitting',
                      'Truck','Cyclist','Pedestrian',
                      'Misc','Van','Tram']
        
        categories.sort()

        self.label_map = {}
        self.label_info = {}
        self.label_info_reverse = {}
        
        counter = 1
        self.label_info[0] = "background"
        
        for c in categories:
            
            self.label_map[counter] = counter
            # {0: 'background'
            self.label_info[counter] = c
            # {'background' : 0 
            self.label_info_reverse[c] = counter
            
            counter += 1
        


    def __getitem__(self, item):
        image, target = super(KittiDataset, self).__getitem__(item) #target 라벨
        image_id = self.images[item].split("\\")[-1][:-4]
        width, height = image.size
        boxes = []
        labels = []
        
        if len(target) == 0:
            return None, None, None, None, None
        
        for annotation in target:
            bbox = annotation.get("bbox")
            
            # left(x), top(y), right(x), bottom(y) -> xmin, ymin, xmax, ymax
            boxes.append([bbox[0] / width,
                          bbox[1] / height,
                          bbox[2] / width,
                          bbox[3] / height])
            
            labels.append(self.label_map[self.label_info_reverse[annotation.get("type")]])
            
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image,
                                                                   (height, width),
                                                                   boxes,
                                                                   labels)
            
        return image, image_id, (height, width), boxes, labels