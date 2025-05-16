''' 
Helper functions for cellpose segmentation and mask saving
'''
import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import cv2 as cv 
import tifffile as tf

def file_sort_key(filename):
  parts = filename.split("-")
  channel = parts[0][-1:] # get the last character of the first part
  location = parts[1]
  return (location,channel)

def plate_location(filename):
  parts = filename.split("-")
  pre_location = parts[1]
  location = pre_location.split(".")[0] # get the first part of the second part
  return location
  
#list all files
def sort_files(dir, image_ext):
  if not dir.exists():
    raise FileNotFoundError("directory does not exist")
  files = sorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name and "SUM" not in f.name],
                           key=lambda x: file_sort_key(x.name))
 # sort by number in filename
  if(len(files)==0):
    raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
  else:
    return files
  
def print_files(files):
  for f in files:
    print(f.name)

def group_files_by_channel(files, nchannels=4):
  grouped = []
  for i in range(0,len(files),nchannels):
    grouped.append(files[i:i+nchannels])
  return grouped

def print_grouped_files(grouped):
  for i in range(len(grouped)):
    print(f"\n Group {i+1} of {len(grouped)}")
    for j in range(len(grouped[i])):
      item = grouped[i][j]
      print(" "+ item.name)
      
def load_image_set(file_group,nchannels=4):
  # load the images from the channels - skip ch3 at position 2 as DAPI is always the last channel
  lastindex = nchannels - 1
  ch1,ch2,ch3 = io.imread(file_group[0]), io.imread(file_group[1]), io.imread(file_group[lastindex])
  image_set = [ch1,ch2*2,ch3]
  return image_set

def get_image_set_name(file_group, index=0):
  # get the name of the image set from the filename
  set_name = plate_location(file_group[index].name)
  return set_name

def save_masks(set_name, masks, image_ext=".tif", mask_type="cell", dir=maskdir):
    # save the masks to a file
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    io.imsave(dir / (set_name + "_" + mask_type + "_masks" + masks_ext), masks)
    
def img_preprocessing(channels):
    from skimage import io, exposure, filters, morphology
    #ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    #channels = [ch1, ch2, ch3]
    for i in range(len(channels)):
        footprint = morphology.disk(5)
        channel = channels[i]
        channel = img_01_normalization(channel)
        channel = exposure.equalize_adapthist(channel, kernel_size=100, clip_limit=0.05)
        #channel = exposure.equalize_hist(channel)
        channel = filters.gaussian(channel, sigma=2)
        
        #channel = filters.median(channel, footprint=footprint)
        #channel = filters.unsharp_mask(channel, radius=1, amount=1)
        
        channels[i] = channel

    multi_channel_image = np.stack(channels, axis=-1)

    #rescaled_image = exposure.rescale_intensity(multi_channel_image, out_range=(0, 255))
    return multi_channel_image
    
    
def img_z_normalization(img):
    # Normalize each channel to z score
    norm_img = (img - np.mean(img)) / np.std(img)
    return norm_img

def img_01_normalization(img):
    # Normalize each channel to the range [0, 1]
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm_img


def img_rescaled(img, factor=0.5):
    from skimage import transform
    rescaled_img = transform.rescale(img, factor, anti_aliasing=False, channel_axis=-1)
    return rescaled_img  

def segment_cell(img, show=True):
    flow_threshold = 0.5
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    diameter = 60

    masks, flows, styles = model.eval(img, batch_size=32, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})
    #plot if true
    if show:
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks
    
def segment_nuclei(orig_img, show=True):
    from skimage import morphology, exposure, filters
    img = orig_img[:,:,2] #get the DAPI channel
    
    img = morphology.closing(img, footprint=morphology.disk(2.5)) #remove small holes

    #remove autofluor
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.min()
    bg = morphology.reconstruction(seed,img,method='dilation')
    img = img - bg
    img = filters.gaussian(img, sigma=1)
    
    flow_threshold = 0.4
    cellprob_threshold = 0
    tile_norm_blocksize = 0
    diameter = None

    masks, flows, styles = model.eval(img, batch_size=32, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})
    if show:
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks

def save_mask_folder(grouped_files_by_channel, image_ext=".tif"):
    for i in trange(len(grouped_files_by_channel)):
        file_group = grouped_files_by_channel[i]
        img_set = load_image_set(file_group)
        img_set_name = get_image_set_name(file_group)
        #print("Set name: ", img_set_name)
        
        stacked_img = img_preprocessing(img_set)
        rescaled_img = img_rescaled(stacked_img, factor=0.25)
        
        cell_masks = segment_cell(rescaled_img, show=False)
        nuc_masks = segment_nuclei(rescaled_img, show=False) 
        
        save_masks(img_set_name, cell_masks, image_ext=image_ext)
        save_masks(img_set_name, nuc_masks, image_ext=image_ext, mask_type="nuclei") 
        

def save_imageJ_masks(set_name, masks, image_ext=".tif", mask_type="cell", dir=maskdir):
    # save the masks to a file
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    masks0 = io.imsave(dir / (set_name + "_" + mask_type + "_masks" + masks_ext))
    io.save_rois(masks0, masks)
    
def preload_and_save_masks(grouped_files_by_channel, outdir, masks_ext=".tif", mask_type="cell"):
    #if you have small images, you may want to load all of them first and then run, so that they can be batched together on the GPU
    print("loading images")
    imgs = load_image_set([grouped_files_by_channel[i] for i in trange(len(grouped_files_by_channel))])

    print("running cellpose-SAM")
    flow_threshold = 0.4
    cellprob_threshold = 0
    tile_norm_blocksize = 0

    masks, flows, styles = model.eval(imgs, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})

    print("saving masks")
    for i in trange(len(grouped_files_by_channel)):
        f = group_files_by_channel[i]
        set_name = get_image_set_name(f)
        io.imsave(outdir / (set_name + mask_type +  "_masks" + masks_ext), masks[i])