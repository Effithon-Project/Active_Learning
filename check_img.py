from torch.utils.data import DataLoader
from torchvision import transforms
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder, kitti_classes
from src.sampler import SubsetSequentialSampler 
from src.dataset import KittiDataset

dboxes = generate_dboxes(model="ssd")
encoder = Encoder(dboxes)

kitti_tot = KittiDataset("D:\\", train=True,
                         transform=SSDTransformer(dboxes, (384, 1280),val=False))

train_loader = DataLoader(kitti_tot)

data = next(iter(train_loader))

img = data[0]
img = img.squeeze()
trans = transforms.ToPILImage()
img_plot = trans(img)
img_plot.show()