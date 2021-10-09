# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os,cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


datasets_path="../../datasets/segBuildings/test/"

for root,dirs,files in os.walk(datasets_path):
    pass

for img in files:

    if img.split('.')[1]=='json':
        continue

    im = cv2.imread(os.path.join(datasets_path,img))

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("./output/", "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    predictor = DefaultPredictor(cfg)
    print(im.shape)



    outputs = predictor(im)
    # print("outputs contains all the information")
    # print(outputs['instances'].to("cpu").get_fields().keys())
    # print(outputs['instances'].to("cpu").get_fields()['pred_boxes'])
    # print(outputs['instances'].to("cpu").get_fields()['scores'])
    # print(outputs['instances'].to("cpu").get_fields()['pred_classes'])
    # print(outputs['instances'].to("cpu").get_fields()['pred_masks'].shape)

    # the folowings are for the visualization.

    from detectron2.utils.visualizer import ColorMode

    balloon_metadata = MetadataCatalog.get("segBuildings_train")
    # balloon_metadata.set(thing_classes=["found_balloon"]) 

    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(img,out.get_image()[:, :, ::-1])
    cv2.waitKey(0)



    cv2.destroyAllWindows()

print("end")