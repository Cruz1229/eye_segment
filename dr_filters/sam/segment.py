import torch
import torchvision
import sys
import os 


import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def __show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def __read_image(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src


def sam_autogenerate(image_file, with_plot=False, gray_scale=False, model="vit_b"):
    image_src = __read_image(image_file=image_file, gray_scale=gray_scale)
        
    if model == "vit_b" :
        sam_checkpoint = os.path.join(os.getcwd(), "..\\diabetic_retinopathy_filters\\\sam\\sam_vit_l_0b3195.pth")
        model_type = "vit_b"
        
    elif model == "vit_l":
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"      
    
    else: 
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    
    device = "cuda"    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_sam =image_src.copy()
    masks = mask_generator.generate(image_src)
    
    if not gray_scale:
        cmap_val = None
    else:
        cmap_val = 'gray'
        
    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        __show_anns(masks)
        ax2.imshow(image_sam, cmap=cmap_val)
        #return True

    return image_sam


def sam_custom(**kwargs):
    """
        image_file, 
        with_plot=False, 
        gray_scale=False, 
        model="vit_b"
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    """
    if not kwargs["image_file"]:
        return 'Error! try using sam_custom(image_file="path_image")'
    
    image_src = __read_image(image_file=kwargs["image_file"], gray_scale=kwargs["gray_scale"] | False)
        
    if kwargs["model"] == "vit_b" :
        sam_checkpoint = "./model/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        
    elif kwargs["model"] == "vit_l":
        sam_checkpoint = "./model/sam_vit_l_0b3195.pth"
        model_type = "vit_l"      
    
    else: 
        sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    
    device = "cuda"    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_sam =image_src.copy()
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side = kwargs["points_per_side"] | 32,
        pred_iou_thresh = kwargs["pred_iou_thresh"] | 0.86,
        stability_score_thresh = kwargs["stability_score_thresh"] | 0.92,
        crop_n_layers = kwargs["crop_n_layers"] | 1,
        crop_n_points_downscale_factor = kwargs["crop_n_points_downscale_factor"] | 2,
        min_mask_region_area = kwargs["min_mask_region_area"] | 100,  # Requires open-cv to run post-processing
    )
    
    masks = mask_generator.generate(mask_generator)
    
    if not kwargs["gray_scale"]:
        cmap_val = None
    else:
        cmap_val = 'gray'
        
    if kwargs["with_plot"]:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        __show_anns(masks)
        ax2.imshow(image_sam, cmap=cmap_val)
        return True

    return image_sam