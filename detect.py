import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os

from matplotlib.image import imread
import scipy.misc
from PIL import Image

from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
register_coco_instances('p_dataset', {}, 'people_dataset/plik.json', 'people_dataset/images')

fruits_nuts_metadata = MetadataCatalog.get("p_dataset")

dataset_dicts = DatasetCatalog.get("p_dataset")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('im',vis.get_image()[:, :, ::-1])
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


 

  
