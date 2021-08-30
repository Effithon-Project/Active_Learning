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
        super(KittiDataset, self).__init__(root, train = True, transform= None)
        
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
        self.label_info_reverse["background"] = 0
        
        for c in categories:
            
            self.label_map[counter] = counter # {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
            self.label_info[counter] = c
            #         {0: 'background',
            #          1: 'Car',
            #          2: 'Cyclist',
            #          3: 'DontCare',
            #          4: 'Misc',
            #          5: 'Pedestrian',
            #          6: 'Person_sitting',
            #          7: 'Tram',
            #          8: 'Truck',
            #          9: 'Van'}
            self.label_info_reverse[c] = counter
            #         {'Car': 1,
            #          'Cyclist': 2,
            #          'DontCare': 3,
            #          'Misc': 4,
            #          'Pedestrian': 5,
            #          'Person_sitting': 6,
            #          'Tram': 7,
            #          'Truck': 8,
            #          'Van': 9}
            
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
#             if image_id == "003382":
#                 print("-"*10)
#                 print(bbox[0])
#                 print(bbox[1])
#                 print(bbox[2])
#                 print(bbox[3])
#                 print("w", width, "h", height)
#                 print(bbox[0]/width)
#                 print(bbox[1]/height)
#                 print(bbox[2]/width)
#                 print(bbox[3]/height)
                
            boxes.append([bbox[0] / width,
                          bbox[1] / height,
                          bbox[2] / width,
                          bbox[3] / height])
            
            labels.append(self.label_map[self.label_info_reverse[annotation.get("type")]])
            
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        if self.transform is not None:
#             print("TRANSFORMED")
            image, (height, width), boxes, labels = self.transform(image,
                                                                   (height, width),
                                                                   boxes,
                                                                   labels)
            
            if image_id == "003382":
                print("Kittidataset")
                print(height, width)
                print(labels.size())
#             print(labels[labels!=0])
        return image, image_id, (height, width), boxes, labels