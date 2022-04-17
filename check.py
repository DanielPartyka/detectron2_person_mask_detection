import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import sys
import datetime
import random
from pathlib import Path
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os

from matplotlib.image import imread
import pylab as plt
import scipy.misc
from PIL import Image

from detectron2.utils.visualizer import ColorMode

from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
register_coco_instances('p_dataset', {}, 'datasets/people_dataset/plik.json', 'datasets/people_dataset/images')

fruits_nuts_metadata = MetadataCatalog.get("p_dataset")

dataset_dicts = DatasetCatalog.get("p_dataset")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ('p_dataset',)
cfg.DATALOADER.NUM_WORKERS = 2
 # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)  
'''
try:
'''	
image_folder_location = sys.argv[1]
path = os.path.realpath(__file__)
file_path = os.path.split(path)
f = file_path[0]
file_path = file_path[0] + '\\' + str(sys.argv[1])
counter = 0
date = str(datetime.datetime.now())
date.replace(" ", "")
x = random.randint(1,20)
output_folder_path = f + '\\' + str(x)
os.mkdir(output_folder_path)

try:

	for image in os.listdir(file_path):
		im = cv2.imread(str(sys.argv[1]) + '\\' + image)
		height, width, channels = im.shape
		outputs = predictor(im)
		v = Visualizer(im[:, :, ::-1],
			   metadata=fruits_nuts_metadata, 
			   scale=0.8, 
			   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
		)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		try:
		  plt.imshow(outputs["instances"].to("cpu").pred_masks.numpy().reshape(height, width), cmap='gray', aspect='auto')
		  plt.axis('off')
		  plt.savefig(output_folder_path + '\\' + image, bbox_inches='tight', pad_inches=0)
		except:
		  if counter == 0:
		  	Path(output_folder_path + '\\' + 'uwagi.txt').touch()
		  file1 = open(output_folder_path + '\\' + 'uwagi.txt', 'a')
		  file1.write('Nie udalo sie utworzyc maski ze zdjecia' + image + '\n')
		  file1.close()
		  counter += 1
  
except:
	print("Wystapil problem z wczytywaniem zdjec")   
  
print('Zapisano zdjecia z maskami do folderu: ', date)
