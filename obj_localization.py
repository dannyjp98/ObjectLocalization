import pandas as pd 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch
import utils
import albumentations as A
from torch import nn 
import timm

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('/content/object-localization-dataset')

"""CONFIGS"""

CSV_FILE = 'object-localization-dataset/train.csv'
DATA_DIR = 'object-localization-dataset/'

DEVICE = 'cuda'
BATCH_SIZE = 16
IMG_SIZE = 140

LR = 0.001
EPOCHS = 40
MODEL_NAME = 'efficientnet_b0'

NUM_COR = 4

df = pd.read_csv(CSV_FILE)

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

"""TEST DATASET"""

row = df.iloc[184]
img = cv2.imread(DATA_DIR + row.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pt1 = (row.xmin, row.ymin)
pt2 = (row.xmax, row.ymax)
bnd_box_img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
plt.imshow(bnd_box_img)

train_df, valid_df = train_test_split(df, test_size = 0.20, random_state = 42)

"""AUGMENTATIONS"""


"""CUSTOM DATASET"""

class ObjLocDataset(torch.utils.data.Dataset):

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax

    bbox = [[xmin, ymin, xmax, ymax]]

    img = cv2.imread(DATA_DIR + row.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.augmentations:
      data = self.augmentations(image = img, bboxes = bbox, class_labels = [None])
      img = data['image']
      bbox = data['bboxes'][0]

    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0 #(h, w, c) -> (c, h, w)
    bbox = torch.Tensor(bbox)

    return img, bbox

"""GENERATE MODEL"""
class ObjLocModel(nn.Module):
  def __init__(self):
    super(ObjLocModel, self).__init__()

    self.backbone = timm.create_model(MODEL_NAME, pretrained = True, num_classes = 4)

  def forward(self, images, gt_bboxes = None):

    bboxes = self.backbone(images)

    if gt_bboxes != None:
      loss = nn.MSELoss()(bboxes, gt_bboxes)
      return bboxes, loss

    return bboxes

"""TRAIN & EVAL FUNCTIONS"""

def train_fn(model, dataloader, optimizer):
  total_loss = 0.0
  model.train() # Dropout ON

  for data in tqdm(dataloader):
    images, gt_bboxes = data
    images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

    bboxes, loss = model(images, gt_bboxes)

    optimizer.zero_grad() 
    loss.backward() #finds gradients
    optimizer.step() #updates weights and biases

    total_loss += loss.item()

  return total_loss/len(dataloader) #avg loss

def eval_fn(model, dataloader):
  total_loss = 0.0
  model.eval() # Dropout ON

  with torch.no_grad():
    for data in tqdm(dataloader):
      images, gt_bboxes = data
      images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

      bboxes, loss = model(images, gt_bboxes)
      total_loss += loss.item()

    return total_loss/len(dataloader) #avg loss


def main():

  """Perform Augmentations"""
  train_augs = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate()
  ], bbox_params=A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels']))

  valid_augs = A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE)
  ], bbox_params=A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels']))

  trainset = ObjLocDataset(train_df, train_augs)
  validset = ObjLocDataset(valid_df, valid_augs)

  print(f"Total examples in the trainset : {len(trainset)}")
  print(f"Total examples in the trainset : {len(validset)}")

  img, bbox = trainset[120]

  xmin, ymin, xmax, ymax = bbox

  pt1 = (int(xmin), int(ymin))
  pt2 = (int(xmax), int(ymax))

  bnd_img = cv2.rectangle(img.permute(1, 2, 0).numpy(),pt1, pt2,(255,0,0),2)
  plt.imshow(bnd_img)

  """# Load dataset into batches"""

  trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
  validloader = torch.utils.data.DataLoader(validset, batch_size = BATCH_SIZE, shuffle = False)

  print("Total no. batches in trainloader : {}".format(len(trainloader)))
  print("Total no. batches in validloader : {}".format(len(validloader)))

  for images, bboxes in trainloader:
    break;
    
  print("Shape of one batch images : {}".format(images.shape))
  print("Shape of one batch bboxes : {}".format(bboxes.shape))


  model = ObjLocModel()
  #model.to(DEVICE);

  random_img = torch.rand(1, 3, 140, 140)#.to(DEVICE)
  model(random_img).shape

  """TRAIN MODEL"""
  optimizer = torch.optim.Adam(model.parameters(), lr = LR)

  best_valid_loss = np.Inf

  for i in range(EPOCHS):

    train_loss = train_fn(model, trainloader, optimizer)
    valid_loss = eval_fn(model, validloader)

    if valid_loss < best_valid_loss:
      torch.save(model.state_dict(), 'best_model.pt')
      print("WEIGHTS ARE SAVED!")
      best_valid_loss = valid_loss

    print(f"Epoch : {i+1} train loss : {train_loss} valid loss : {valid_loss}")


  model.load_state_dict(torch.load('best_model.pt'))
  model.eval()

  with torch.no_grad():
    image, gt_bbox = validset[14] #(c, h, w)
    image = image.unsqueeze(0).to(DEVICE) #(bs, c, h, w) At this point our image is (c,h,w) but our image should have batchsize.
    out_bbox = model(image)

    utils.compare_plots(image, gt_bbox, out_bbox)

if __name__ == "__main__":
  main()