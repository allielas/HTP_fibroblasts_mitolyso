"""
Helper functions for cellpose segmentation and mask saving
Allie Spangaro, Toronto Metropolitan University
"""

import numpy as np
from cellpose import models, core, io, plot, utils
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tf
import re
from skimage import exposure, filters, morphology, transform


def file_sort_key(filename):
    """
    Generate a key to sort a list of image files in the 'MAX_chN-rXXcYYfZZ.tif' filename nomenclature by their plate location and channel.
    > e.g: MAX_ch1-r02c02f01.tif, MAX_ch2-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch1-r02c02f02.tif, MAX_ch2-r02c02f02.tif, MAX_ch3-r02c02f02.tif, MAX_ch3-r02c02f02.tif
    Parameters:
          filename (str): the filename from the image path
    Returns:
          (location,channel) (tuple of str): the list of files sorted by location and channel. Images will be ordered by location first, and then by channel to match the order seen in CellProfiler
    """
    parts = filename.split("-")
    channel = parts[0][
        -1:
    ]  # get the last character of the first part for the channel number
    location = parts[1]  # get the rXXcYYfZZ.tif portion
    return (location, channel)


def get_location_code(file_name):
    """
    Parse a string from an image filename in the MAX_chN-rXXcYYfZZ.tif filename nomenclature
    Parameters:
          filename (str): the filename from the image path
    Returns:
          location (str): the rowcolfield plate location of the image
    """
    match = re.search(r"(r\d{2}c\d{2}f\d{2})", file_name)
    if match:
        # print(match.group(0))  # Print the full match for debugging
        return match.group(1)
    else:
        return None


def get_plate_number(file_name):
    """
    Parse a string from an image filename in the rep0N_MAX_chN-rXXcYYfZZ.tif filename nomenclature to get the plate number
    Parameters:
          filename (str): the filename from the image path
    Returns:
          plate_number (int): the plate number of the image
    """
    match = re.search(r"rep(\d{1,2})", file_name)
    if match:
        return int(match.group(1))
    else:
        return None


def sort_files(dir, image_ext):
    """
    Sort the list of directories sorted by their plate location and channel
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          files (list of Path objects): the list of files sorted by location and channel
    """
    if not dir.exists():
        raise FileNotFoundError("directory does not exist")
    files = sorted(
        [
            f
            for f in dir.glob("*" + image_ext)
            if "_masks" not in f.name and "_flows" not in f.name and "SUM" not in f.name
        ],
        key=lambda x: file_sort_key(x.name),
    )
    # sort by number in filename
    if len(files) == 0:
        raise FileNotFoundError(
            "no image files found, did you specify the correct folder and extension?"
        )
    else:
        return files


def print_files(files):
    """
    Print a list of filenames from a list of Path objects
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    for f in files:
        print(f.name)


def load_sorted_directory_list(dir, image_ext=".tif"):
    """
    Load the list of directories sorted by their location and channel to allow grouping into a 2D list and force the objects to be Path objects
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          file_list (list of Path objects): the list of files sorted by location and channel
    """
    dir = Path(dir)
    file_list = sort_files(dir, image_ext)
    return file_list


def get_nchannels(ordered_files):
    """
    Get the number of channels based on the last element list element in a list of image files ordered by channels
    Parameters:
          ordered_files (1D list of Path objects): list of files grouped into image sets by their channels
    Returns:
          nchannels (int): the number of channels based on the length of the first element
    """
    lastitem = ordered_files[-1].name
    parts = lastitem.split("-")
    ch_char = parts[0][-1:]
    nchannels = int(ch_char)
    return nchannels


def get_image_set_name(grouped_files_by_channel, index=1):
    """
    Get the name of the image set from the specified index in a list of filenames grouped by channel
    Corresponds to image set index in cellprofiler based on the order in the folder
    Parameters:
          grouped_files_by_channel (2D list of Path objects): an ordered 2D list of file paths grouped by channel and ordered by platemap location
          index (int, optional): the index to look up the rowcolfield of that image set
    Returns:
          set_name (str): String name of the image set from the rowcolfield filename nomenclature
    """
    if isinstance(grouped_files_by_channel, dict):
        grouped_files_by_channel = list(grouped_files_by_channel.values())
    set_name = f"rep{str(get_plate_number(grouped_files_by_channel[index - 1].as_posix())).rjust(2, '0')}_{get_location_code(grouped_files_by_channel[index - 1].name)}"
    return set_name


def group_files_by_channel(files, nchannels=None):
    """
    Load the list of directories
    Parameters:
          dir (Path object or str): the directory containing the images
          nchannels (int,optionsl): the number of image channels to use for grouping
    Returns:
          grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    grouped_files_by_channel = []
    if nchannels == None:
        nchannels = get_nchannels(files)

    for i in range(0, len(files), nchannels):
        grouped_files_by_channel.append(files[i : i + nchannels])
    return grouped_files_by_channel


def group_files_by_channel_dict(files, nchannels=None):
    """
    Group image files by channel and return a dictionary keyed by image set name.
    Parameters:
          files (list of Path objects): the list of files sorted by location and channel
          nchannels (int, optional): the number of image channels to use for grouping
    Returns:
          grouped_files_by_channel (dict): dictionary with image set names as keys and nchannels-long file lists as values
    """
    grouped_files_by_channel = {}
    if nchannels is None:
        nchannels = get_nchannels(files)

    for i in range(0, len(files), nchannels):
        file_group = files[i : i + nchannels]
        set_name = get_image_set_name(file_group)
        grouped_files_by_channel[set_name] = file_group
    return grouped_files_by_channel


def print_grouped_files(grouped_files_by_channel):
    """
    Print a list of files grouped by channel to confirm that the images are loaded in the correct order
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    for i in range(len(grouped_files_by_channel)):
        if isinstance(grouped_files_by_channel, dict):
            print(
                f"\n Group {i + 1} of {len(grouped_files_by_channel)}: {list(grouped_files_by_channel.keys())[i]}"
            )
            for j in range(
                len(grouped_files_by_channel[list(grouped_files_by_channel.keys())[i]])
            ):
                item = grouped_files_by_channel[
                    list(grouped_files_by_channel.keys())[i]
                ][j]
                print(" " + item.name)
        else:
            print(f"\n Group {i + 1} of {len(grouped_files_by_channel)}")
            for j in range(len(grouped_files_by_channel[i])):
                item = grouped_files_by_channel[i][j]
                print(" " + item.name)


def load_image_set(single_grouped_files_by_channel, nchannels=None):
    """
    Load an image set given file paths for a single image set; load images from a single element of the list made by the `group_files_by_channel` function
    Parameters:
          single_grouped_files_by_channel (list of Path objects): list of file paths grouped by channel and ordered by platemap location
          nchannels (int or None): the number of channels. Assumes based on length of first element otherwise
    Returns:
          image_set (list of 2D arrays): a list of 2D arrays representing the loaded single-channel grayscale images
    """
    if nchannels == None:
        nchannels = len(single_grouped_files_by_channel)
    # load the images from the channels - skip ch3 at position 2 as DAPI is always the last channel
    lastindex = nchannels - 1  # subtract 1 to convert to the 0-index
    ch1, ch2, ch3 = (
        io.imread(single_grouped_files_by_channel[0]),
        io.imread(single_grouped_files_by_channel[1]),
        io.imread(single_grouped_files_by_channel[lastindex]),
    )
    image_set = [ch1, ch2 * 2, ch3]
    return image_set


def get_image_set_without_modifications(
    single_grouped_files_by_channel, selected_channels=[], nchannels=None
):
    """
    Load an image set given file paths for a single image set; load images from a single element of the list made by the `group_files_by_channel` function
    Parameters:
          single_grouped_files_by_channel (list of Path objects): list of file paths grouped by channel and ordered by platemap location
          selected_channels (list of int): a list with the desired channels to use/process (1-indexed)
          nchannels (int or None): the number of channels. Assumes based on length of first element otherwise
    Returns:
          image_set (list of 2D arrays): a list of 2D arrays representing the loaded single-channel grayscale images
    """
    # can choose to load only a subset of channels if desired, otherwise will load all channels
    if selected_channels == []:
        nchannels = len(single_grouped_files_by_channel)
    else:
        nchannels = len(selected_channels)
    # load the images from the channels - skip ch3 at position 2 as DAPI is always the last channel

    image_set = []
    for i in range(nchannels):
        image_channel = io.imread(single_grouped_files_by_channel[i])
        image_set.append(image_channel)
    return image_set


def get_multichannel_img_normalized(img_set, selected_channels=[1, 2, 3, 4]):
    """
    Create a multichannel grayscale image given an list of single-channel grayscale images and stack the image together
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
    Returns:
          multi_channel_image (3D array): an array containing the grayscale images with channel in the third dimension
    """
    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    for chnum in selected_channels:
        # footprint = morphology.disk(5)
        channel = img_set[chnum - 1]
        channel = img_01_normalization(channel)
        img_stack.append(channel)

    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def img_preprocessing(img_set):
    """
    Preprocess a grayscale image given an list of single-channel grayscale images and stack the image together
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
    Returns:
          multi_channel_image (3D array): a 3D array containing the preprocessed grayscale images
    """
    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    for channel in img_set:
        # footprint = morphology.disk(5)
        channel = img_01_normalization(channel)
        channel = exposure.equalize_adapthist(channel, kernel_size=100, clip_limit=0.05)
        channel = filters.gaussian(channel, sigma=2)
        img_stack.append(channel)

    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def img_preprocessing_v2(img_set):
    """
    Preprocess a grayscale image given an list of single-channel grayscale images with historam equalization, and median filter smoothing and stack the image together
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
    Returns:
          multi_channel_image (3D array): a 3D array containing the preprocessed grayscale images
    """
    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    for channel in img_set:
        # footprint = morphology.disk(5)
        # channel = img_01_normalization(channel)
        channel = exposure.equalize_adapthist(channel, kernel_size=100, clip_limit=0.02)
        channel = filters.median(channel, morphology.disk(2))
        img_stack.append(channel)

    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def stack_images_with_histogram_matching(
    img_set, selected_channels=[1, 2, 4], reference_channel=2
):
    """
    Stack and preprocess multiple grayscale images given a list of single-channel grayscale images given a referene channel for matching
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
           selected channels (list of int): a list with the desired channels to use/process (1-indexed)
           reference_channel (int, optional): the channel to use as a reference for histogram matching, 2 by default
    Returns:
          multi_channel_image (3D array): a 3D array containing the preprocessed grayscale images
    """
    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    for i, image in enumerate(img_set):
        channel_number = i + 1  # convert to 1-indexed
        channel_image = image
        if channel_number in selected_channels:
            if channel_number != reference_channel:
                channel_image = exposure.match_histograms(
                    image=channel_image,
                    reference=img_set[reference_channel - 1],
                    channel_axis=None,
                )
            img_stack.append(channel_image)

    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def img_zscore_normalization(img):
    """
    ### Preprocess a grayscale image by normalizing pixels to a standard-scaled distribution (mean of 0, std of 1)
    Recommented for 2D only
    Parameters:
          img (2D or 3D array): a grayscale image as a 2D array
    Returns:
          norm_img (2D or 3D array): the grayscale image array normalized by z score
    """
    # Normalize each channel to z score
    norm_img = (img - np.mean(img)) / np.std(img)
    return norm_img


def img_01_normalization(img):
    """
    Normalize the intensity of each channel in a 16-bit grayscale image to the range [0, 1] to aid in preprocessing or segmentation by some algorithms
    Parameters:
          img (2D or 3D array): a grayscale image as a 2D array
    Returns:
          norm_img (2D or 3D array): the grayscale image array normalized by to [0,1]
    """
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm_img


def img_rescaled(img, factor=0.5, anti_aliasing=False, channel_axis=-1):
    """
    Rescale a multichannel grayscale image by a given factor using skimage.transform.rescale
    Parameters:
          img (3D array): 3D array containing the grayscale image stack
          factor (float, optional): Factor to rescale the image by, 50% by default
          anti_aliasing (bool, optional): Flag whether to downsample the image or not, will downsample by default to reduce noise
    Returns:
          rescaled_image (3D array): a 3D array containing the rescaled grayscale image
    """

    rescaled_img = transform.rescale(
        img, factor, anti_aliasing=anti_aliasing, channel_axis=channel_axis
    )
    return rescaled_img


def load_model(pretrained_model="cpsam_v2", gpu=True):
    """
    Loads a cellpose model, uses cp_sam by default but can specify any model
    Parameters:
          pretrained_model (None or str): the name of the pretrained model specified, by default assumes cpsam
          gpu (bool, optional): specify whether or not to use gpu, true by default

    Returns:
          model (Cellpose.model object): the model loaded
    """
    io.logger_setup()  # run this to get printing of progress
    if gpu == True:
        if core.use_gpu() == False:
            raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(pretrained_model=pretrained_model, gpu=True)
    return model


def spline_order(interpolation_method: str) -> int:
    """Determine interpolation order from method.
    Args:
        interpolation_method (str): The interpolation method to use. Must be one of 'nearest', 'bilinear', or 'bicubic'.
    Returns:
        int: The interpolation order.
    """
    interpolation_method = interpolation_method.lower()
    if interpolation_method == "nearest":
        return 0

    elif interpolation_method == "bilinear":
        return 1

    elif interpolation_method == "bicubic":
        return 3

    else:
        raise ValueError(
            f"Invalid interpolation method: {interpolation_method}. Must be one of 'nearest', 'bilinear', or 'bicubic'."
        )


def preprocessing_for_cell_segmentation(
    orig_img,
    composite_channels=[1, 2],
    nucleus_channel=3,
    rescale_factor=0.15,
    interpolation_method="bicubic",
    anti_aliasing=False,
):
    """
    Preprocess a grayscale image of the cell channels, run cellpose on the image, and return the predicted masks
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           nucleus_channel (int, optional): the channel to use for the nuclear channel, 3 by default
    Returns:
          preprocessed_img (2D or 3D array): the preprocessed grayscale image for cell segmentation
    """
    # combine the specified channels to help cellpose out a bit
    gfp = orig_img[:, :, composite_channels[0] - 1]
    rfp = orig_img[:, :, composite_channels[1] - 1]
    # save the specified nuclear channel for later
    dapi = orig_img[:, :, nucleus_channel - 1]
    # add the channels and normalize intensity
    img_combo = gfp + rfp
    img_combo = img_01_normalization(img_combo)
    img_combo = exposure.equalize_adapthist(
        img_combo, kernel_size=256, clip_limit=0.02, nbins=256
    )
    dapi = img_01_normalization(dapi)
    dapi = exposure.equalize_adapthist(
        dapi, kernel_size=128, clip_limit=0.02, nbins=256
    )
    # denoise / smooth with gaussian kernel
    img_combo = filters.gaussian(img_combo, sigma=2)
    dapi = filters.gaussian(dapi, sigma=3)

    # stack the images
    preprocessed_img = np.stack([img_combo, dapi], axis=-1)
    # rescale the images for speed
    preprocessed_img = transform.rescale(
        preprocessed_img,
        scale=rescale_factor,
        order=spline_order(interpolation_method),
        mode="symmetric",
        channel_axis=-1,
        anti_aliasing=anti_aliasing,
    )
    return preprocessed_img


def preprocessing_for_nuclei_segmentation(
    orig_img,
    nucleus_channel=3,
    gaussian_sigma=2,
    rescale_factor=0.33,
    interpolation_method="bicubic",
    anti_aliasing=False,
    use_adaptive_histogram_equalization=True,
    use_white_tophat=True,
    use_median_filter=False,
    use_closing=True,
    use_unsharp_mask=True,
):
    """
    Preprocess a grayscale image of the nuclear channel, run cellpose on the image, and return the predicted masks
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           nucleus_channel (int, optional): the channel to use for the nuclear channel, 3 by default
           gaussian_sigma (float, optional): the sigma for the gaussian kernel used to smooth the image, 2 by default
           rescale_factor (float, optional): the factor to rescale the image by, 0.25 by default
           interpolation_method (str, optional): the interpolation method to use for rescaling, 'bicubic' by default
              anti_aliasing (bool, optional): flag whether to use anti-aliasing when rescaling, false by default
              use_white_tophat (bool, optional): flag whether to use white top-hat filtering to remove background, true by default
              use_unsharp_mask (bool, optional): flag whether to use unsharp masking to sharpen the image, false by default

    Returns:
          preprocessed_img (2D or 3D array): the preprocessed grayscale image for nuclear segmentation
    """
    dapi = orig_img[:, :, nucleus_channel - 1]
    dapi = img_01_normalization(dapi)
    if use_adaptive_histogram_equalization:
        dapi = exposure.equalize_adapthist(
            dapi, kernel_size=100, clip_limit=0.01, nbins=512
        )
    if use_median_filter:
        dapi = filters.median(dapi, morphology.disk(2))
    if use_white_tophat:
        bg2 = morphology.white_tophat(dapi, morphology.disk(3))
        dapi = dapi - bg2
    if use_closing:
        dapi = morphology.closing(dapi, morphology.disk(2.5))
        dapi = img_01_normalization(dapi)
    if use_unsharp_mask:
        # sharpen image and improve outline (radius is the gaussian kernel)
        dapi = filters.unsharp_mask(dapi, radius=gaussian_sigma, amount=1)
    else:
        dapi = filters.gaussian(dapi, sigma=gaussian_sigma)

    # rescale the images for speed
    preprocessed_img = transform.rescale(
        dapi,
        scale=rescale_factor,
        order=spline_order(interpolation_method),
        mode="symmetric",
        anti_aliasing=anti_aliasing,
    )
    return preprocessed_img


def run_cellpose_segmentation(
    img,
    model,
    show_plot=True,
    flow_threshold=0.4,
    cellprob_threshold=0,
    tile_norm_blocksize=100,
    diameter=None,
    min_size=100,
    max_size_frac=0.80,  # keep masks up to 80% of image size
    niter=1000,
    dilate_radius=0,
    fill_holes=True,
    show_image_preprocessing=False,
    save_plots=False,
    plot_dir="",
    save_name="cell_segmentation_result",
):
    """
    Run cellpose-SAM on a grayscale multichannel cell image and return the predicted masks.
    Trained on cells ranging from 7.5-120 pixels, so rescale accordingly
    Parameters:
            img (2D or 3D array): grayscale image to be segmented by cellpose. ideally preprocessed already
            model (Cellpose.model): the cellpose model used for segmentation
            show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
            show_image_preprocessing (bool, optional): flag whether to show the preprocessed image
            flow_threshold (float, optional): the flow threshold for cellpose, (from original 0.4 default). Down for more stringent, up for more lenient
            cellprob_threshold (float, optional): the cell probability threshold for cellpose, 0 by default (from original 0 default).
            tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
            diameter (int or None, optional): the diameter for cellpose, 60 as experimentally determined, but None by default
            min_size (int, optional): the minimum size of masks to keep, 100 pixels by default
            max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.85 by default
            niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
            masks (list of 2D or 3D arrays): the predicted masks from the cellpose model
    """
    if show_image_preprocessing:
        tf.imshow(img, cmap="plasma")

    masks, flows, styles = model.eval(
        img,
        batch_size=64,
        niter=niter,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
        min_size=min_size,
    )
    if fill_holes:
        masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
    if dilate_radius > 0:
        masks = utils.dilate_masks(masks, n_iter=dilate_radius)
    # plot if true
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        if save_plots:
            save_dir = Path(f"{plot_dir}/segmentation_plots")
            Path.mkdir(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir.as_posix()}/{save_name}.png")
            # print(f"Saved plot to {save_dir.as_posix()}/{save_name}.png")
        else:
            plt.show()
    return masks


def segment_cell(img, model, show_plot=True):
    """
    Run cellpose on a grayscale cell image and return the predicted masks
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show (bool, optional): flag whether to show a plot of the predicted mask flow
    Returns:
          masks (list of 2D or 3D arrays): the predicted masks from the cellpose model
    """
    flow_threshold = 0.5
    cellprob_threshold = -1
    tile_norm_blocksize = 0
    diameter = 60

    masks, flows, styles = model.eval(
        img,
        batch_size=32,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )
    # plot if true
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks


def segment_nuclei(orig_img, model, show_plot=True):
    """
    Preprocess a grayscale image of the nuclear channel, run cellpose on the image, and return the predicted masks
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show (bool, optional): flag whether to show a plot of the predicted mask flow

    Returns:
          masks (list of 2D or 3D arrays): the predicted nuclear masks from the cellpose model
    """
    img = orig_img[:, :, 2]  # get the DAPI channel

    # remove background
    dog = filters.difference_of_gaussians(img, low_sigma=2.5)
    seed = np.minimum(dog, img)  # ensure seed is not greater than the original image
    bg = morphology.reconstruction(seed, img, method="dilation")
    img = img - bg

    # remove speckle-shaped autofluor
    bg2 = morphology.white_tophat(img, morphology.disk(3))
    img = img - bg2
    img = morphology.closing(img, morphology.disk(2.5))
    img = filters.gaussian(img, sigma=1)

    flow_threshold = 0.5
    cellprob_threshold = 0
    tile_norm_blocksize = 0
    diameter = None

    masks, flows, styles = model.eval(
        img,
        batch_size=32,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks


def segment_cell_v2(
    img,
    model,
    show_plot=True,
    flow_threshold=0.6,
    cellprob_threshold=-1,
    tile_norm_blocksize=100,
    diameter=60,
    min_size=500,
    max_size_frac=0.85,  # keep masks up to 70% of image size
    niter=1000,
):
    """
    Run cellpose-SAM on a grayscale multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           flow_threshold (float, optional): the flow threshold for cellpose, 0.6 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, -1 by default (from original 0 default).
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, 60 as experimentally determined, but None by default
           min_size (int, optional): the minimum size of masks to keep, 500 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.85 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted masks from the cellpose model
    """
    gfp = img[:, :, 0]  # combine the ch1 and ch2 images to help cellpose out a bit
    rfp = img[:, :, 1]
    dapi = img[:, :, 2]  # save ch3 for later

    img_combo = gfp + rfp
    img_combo = img_01_normalization(img_combo)
    # tf.imshow(img_combo, cmap="viridis")
    # smooth image and improve outline (sigma is the gaussian kernel)
    img_combo = filters.gaussian(img_combo, sigma=1)
    # Can also use unsharp mask, but it tends to chop the outlines too short img_combo = filters.unsharp_mask(img_combo, radius=0.5, amount=2)

    # stack the images
    img_selected_channels = np.stack([img_combo, dapi], axis=-1)

    masks, flows, styles = model.eval(
        img_selected_channels,
        batch_size=64,
        niter=niter,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
        min_size=min_size,
    )
    masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
    masks = utils.dilate_masks(masks, n_iter=2)
    # plot if true
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks


def segment_nuclei_v2(
    orig_img,
    model,
    show_plot=True,
    flow_threshold=0.5,
    cellprob_threshold=1,
    tile_norm_blocksize=100,
    min_size=400,
    max_size_frac=0.4,
    diameter=None,
    niter=None,
):
    """
    Run cellpose-SAM on the nucleus channel from a multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           flow_threshold (float, optional): the flow threshold for cellpose, 0.5 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, 1 by default (from original 0 default). Up to be more stringent, down for a more lenient threshold
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, None by default
           min_size (int, optional): the minimum size of masks to keep, 400 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.4 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted nuclei masks from the cellpose model
    """
    img = orig_img[:, :, 2]  # get the DAPI channel

    # remove speckle-shaped autofluor
    bg2 = morphology.white_tophat(img, morphology.disk(3))
    img = img - bg2
    img = morphology.closing(img, morphology.disk(2.5))

    # sharpen image and improve outline (radius is the gaussian kernel)
    img = img_01_normalization(img)
    img = filters.gaussian(img, sigma=2)
    # tf.imshow(img, cmap="plasma")
    # img = filters.unsharp_mask(img, radius=1, amount=2)
    masks, flows, styles = model.eval(
        img,
        batch_size=64,
        diameter=diameter,
        niter=niter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
    )
    # dilate before removing the ones touching edges to catch the stragglers
    masks = utils.dilate_masks(masks, n_iter=1)
    masks_removed_edges = utils.remove_edge_masks(masks)
    masks_removed_edges = utils.fill_holes_and_remove_small_masks(
        masks_removed_edges, min_size=min_size
    )

    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks_removed_edges, flows[0])
        plt.tight_layout()
        plt.show()
    return masks_removed_edges


def segment_nuclei_v3(
    orig_img,
    model,
    nucleus_channel=3,
    show_plot=True,
    flow_threshold=0.5,
    cellprob_threshold=0,
    tile_norm_blocksize=0,
    min_size=400,
    max_size_frac=0.4,
    diameter=None,
    niter=None,
    show_image_preprocessing=False,
    save_plots=False,
    plot_dir="",
    save_name="nuclei_segmentation_result",
):
    """
    Run cellpose-SAM on the nucleus channel from a multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           flow_threshold (float, optional): the flow threshold for cellpose, 0.5 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, 0 by default (from original 0 default). Up to be more stringent, down for a more lenient threshold
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, None by default
           min_size (int, optional): the minimum size of masks to keep, 400 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.4 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted nuclei masks from the cellpose model
    """
    img = orig_img[:, :, nucleus_channel - 1]  # get the DAPI channel (and 0-index it)
    # img = img_01_normalization(img)

    # #do a rolling ball background subtraction
    # from skimage import data, restoration, util
    # background = restoration.rolling_ball(
    #     img, kernel=restoration.ellipsoid_kernel((25, 25), 0.1)
    # )
    # img = img - background
    # img = img_01_normalization(img)
    # # plot_result(img, background)
    # # plt.show()
    # img = filters.gaussian(img, sigma=1)

    # remove speckle-shaped autofluor
    bg2 = morphology.white_tophat(img, morphology.disk(3))
    img = img - bg2
    img = morphology.closing(img, morphology.disk(2.5))

    # sharpen image and improve outline (radius is the gaussian kernel)
    img = img_01_normalization(img)
    img = filters.gaussian(img, sigma=2)
    if show_image_preprocessing:
        tf.imshow(img, cmap="plasma")
    masks, flows, styles = model.eval(
        img,
        batch_size=64,
        diameter=diameter,
        niter=niter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
    )
    # dilate before removing the ones touching edges to catch the stragglers
    masks = utils.dilate_masks(masks, n_iter=1)
    masks_removed_edges = utils.remove_edge_masks(masks)
    masks_removed_edges = utils.fill_holes_and_remove_small_masks(
        masks_removed_edges, min_size=min_size
    )
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks_removed_edges, flows[0])
        plt.tight_layout()
        if save_plots:
            save_dir = Path(f"{plot_dir}/segmentation_plots")
            Path.mkdir(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir.as_posix()}/{save_name}.png")
            # print(f"Saved plot to {save_dir.as_posix()}/{save_name}.png")
        else:
            plt.show()
    return masks_removed_edges


def segment_cell_v3(
    img,
    model,
    show_plot=True,
    flow_threshold=0.5,
    cellprob_threshold=-1,
    tile_norm_blocksize=100,
    diameter=None,
    min_size=500,
    max_size_frac=0.70,  # keep masks up to 70% of image size
    niter=1000,
    show_image_preprocessing=False,
    save_plots=False,
    plot_dir="",
    save_name="cell_segmentation_result",
):
    """
    Run cellpose-SAM on a grayscale multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           show_image_preprocessing (bool, optional): flag whether to show the preprocessed image
           flow_threshold (float, optional): the flow threshold for cellpose, 0.6 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, -1 by default (from original 0 default).
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, 60 as experimentally determined, but None by default
           min_size (int, optional): the minimum size of masks to keep, 500 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.85 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted masks from the cellpose model
    """
    gfp = img[:, :, 0]  # combine the ch1 and ch2 images to help cellpose out a bit
    rfp = img[:, :, 1]
    dapi = img[:, :, 2]  # save ch3 for later

    img_combo = gfp + rfp
    img_combo = img_01_normalization(img_combo)
    # tf.imshow(img_combo, cmap="viridis")
    # smooth image and improve outline (sigma is the gaussian kernel)
    img_combo = filters.gaussian(img_combo, sigma=1)
    # Can also use unsharp mask, but it tends to chop the outlines too short img_combo = filters.unsharp_mask(img_combo, radius=0.5, amount=2)

    # stack the images
    img_selected_channels = np.stack([img_combo, dapi], axis=-1)
    if show_image_preprocessing:
        plot = microfilm_plot(img_selected_channels)
    masks, flows, styles = model.eval(
        img_selected_channels,
        batch_size=64,
        niter=niter,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
        min_size=min_size,
    )
    masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
    masks = utils.dilate_masks(masks, n_iter=2)
    # plot if true
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
        plt.tight_layout()
        if save_plots:
            save_dir = Path(f"{plot_dir}/segmentation_plots")
            Path.mkdir(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir.as_posix()}/{save_name}.png")
            # print(f"Saved plot to {save_dir.as_posix()}/{save_name}.png")
        else:
            plt.show()
    return masks


def save_masks(
    set_name, masks, outdir, image_ext=".tif", mask_type="cell", model_name=None
):
    """
    Save masks from a previously run cellpose model to a folder given an output directory
    Parameters:
           set_name (str): the image set name to be saved in the output file
           masks (list of 2D or 3D arrays): the masks output from Cellpose
           outdir (Path or str): the output directory to save masks
           image_ext (str, optional): image extension, .tif by default
           mask_type (str, optional): specify the mask type to save, cell by default
           model_name (str, optional): the name of the model used for saving, None by default
    """
    # save the masks to a file
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    if model_name:
        io.imsave(
            outdir
            / (set_name + "_" + mask_type + "_" + model_name + "_masks" + masks_ext),
            masks,
        )
    else:
        io.imsave(outdir / (set_name + "_" + mask_type + "_masks" + masks_ext), masks)


def save_mask_folder_v2(
    ordered_files,
    outdir,
    image_ext=".tif",
    nchannels=None,
    rescale_factor_cell=0.15,
    rescale_factor_nuc=0.33,
    histogram_matching=True,
    reference_channel=2,
    nucleus_channel=3,
    pretrained_model="cpsam_v2",
    gpu=True,
    show_model_name=False,
    save_plots=False,
    selected_channels=[1, 2, 4],
):
    """
    Run cellpose and save cell and nuclear masks to a folder given an ordered list of files ordered by channel and an output directory
    Parameters:
          ordered_files (list): an ordered list of image file paths to be processed and saved; assumed to be ordered by location,channel by the `load_sorted_directory_list`
          outdir (Path or str): the output directory to save masks
          image_ext (str, optional): image extension, .tif by default
          nchannels (int, optional): specify the number of channels, default from the number of elements in the 1st element of the 2D list given
          rescale_factor (float): the factor to rescale the image by for cellpose, 512x512 by default
          pretrained_model (str, optional): the name of the pretrained Cellpose model specified, by default assumes cpsam
          gpu (bool, optional): specify whether or not to use gpu, true by default
          show_model_name (bool, optional): flag whether to show the model name in the saved mask file name, false by default
          save_plots (bool, optional): flag whether to save the segmentation plots, false by default
    """
    # first load model and initialize chanels if not specified
    model = load_model(pretrained_model=pretrained_model, gpu=gpu)
    if nchannels == None:  # handle default case when nchannels isn't specified
        nchannels = get_nchannels(ordered_files)
    grouped_files_by_channel = group_files_by_channel(ordered_files, nchannels)

    # loop through the grouped files and stack /preprocess the images
    for i in trange(len(grouped_files_by_channel)):
        file_group = grouped_files_by_channel[i]
        img_set_name = get_image_set_name(file_group)
        this_nchannels = nchannels
        if "rep04" in img_set_name:
            # for this one set, we only have 3 channels, so we need to specify that
            selected_channels = [1, 2, 3]
            this_nchannels = len(selected_channels)
        img_set_list = get_image_set_without_modifications(
            file_group, nchannels=this_nchannels
        )
        if histogram_matching:
            stacked_img = stack_images_with_histogram_matching(
                img_set_list,
                reference_channel=reference_channel,
                selected_channels=selected_channels,
            )
        else:
            stacked_img = img_preprocessing_v2(img_set_list)

        # Now preprocess the images
        cell_preproessed = preprocessing_for_cell_segmentation(
            stacked_img,
            rescale_factor=rescale_factor_cell,
            nucleus_channel=nucleus_channel,
        )
        nucleus_preproessed = preprocessing_for_nuclei_segmentation(
            stacked_img,
            rescale_factor=rescale_factor_nuc,
            nucleus_channel=nucleus_channel,
        )

        # prepare variables for saving plots and masks
        model_name = pretrained_model if show_model_name else ""
        plot_dir = ""
        save_name_nuc = ""
        save_name_cell = ""
        if save_plots:
            plot_dir = Path(f"{outdir}/segmentation_plots")
            Path.mkdir(plot_dir, exist_ok=True)
            save_name_nuc = f"{img_set_name}_nuclei_{model_name}_segmentation_result"
            save_name_cell = f"{img_set_name}_cell_{model_name}_segmentation_result"
        # run cellpose and save masks
        cell_masks = run_cellpose_segmentation(
            cell_preproessed,
            model,
            diameter=None,
            cellprob_threshold=0,
            flow_threshold=0.4,
            show_plot=save_plots,
            save_plots=save_plots,
            plot_dir=plot_dir,
            save_name=save_name_cell,
        )
        nuc_masks = run_cellpose_segmentation(
            nucleus_preproessed,
            model,
            diameter=None,
            cellprob_threshold=0,
            flow_threshold=0.4,
            show_plot=save_plots,
            save_plots=save_plots,
            plot_dir=plot_dir,
            save_name=save_name_nuc,
        )
        # Now save the masks to the output directory
        save_masks(
            img_set_name,
            cell_masks,
            outdir,
            image_ext=image_ext,
            mask_type="cell",
            model_name=model_name,
        )
        save_masks(
            img_set_name,
            nuc_masks,
            outdir,
            image_ext=image_ext,
            mask_type="nuclei",
            model_name=model_name,
        )


def save_mask_folder(
    ordered_files,
    outdir,
    image_ext=".tif",
    nchannels=None,
    resize_factor=0.25,
    v2=True,
    pretrained_model="cpsam_v2",
    gpu=True,
    show_model_name=False,
    save_plots=False,
):
    """
    Run cellpose and save cell and nuclear masks to a folder given an ordered list of files ordered by channel and an output directory
    Parameters:
          ordered_files (list): an ordered list of image file paths to be processed and saved; assumed to be ordered by location,channel by the `load_sorted_directory_list`
          outdir (Path or str): the output directory to save masks
          image_ext (str, optional): image extension, .tif by default
          nchannels (int, optional): specify the number of channels, default from the number of elements in the 1st element of the 2D list given
          resize_factor (float): the factor to rescale the image by for cellpose, 512x512 by default
    """
    model = load_model(pretrained_model=pretrained_model, gpu=gpu)
    if nchannels == None:  # handle default case when nchannels isn't specified
        nchannels = get_nchannels(ordered_files)

    grouped_files_by_channel = group_files_by_channel(ordered_files, nchannels)

    for i in trange(len(grouped_files_by_channel)):
        file_group = grouped_files_by_channel[i]
        img_set_name = get_image_set_name(file_group)
        this_nchannels = nchannels
        if "rep04" in img_set_name:
            this_nchannels = 3  # for this one set, we only have 3 channels, so we need to specify that
        img_set = load_image_set(file_group, this_nchannels)
        # print("Set name: ", img_set_name)
        if v2:
            stacked_img = img_preprocessing_v2(img_set)
        else:
            # old function
            stacked_img = img_preprocessing(img_set)

        # rescale to 512 by 512 for processing speed
        rescaled_img = img_rescaled(stacked_img, factor=resize_factor)

        model_name = pretrained_model if show_model_name else ""
        plot_dir = ""
        save_name_nuc = ""
        save_name_cell = ""
        if save_plots:
            plot_dir = Path(f"{outdir}/segmentation_plots")
            Path.mkdir(plot_dir, exist_ok=True)
            save_name_nuc = f"{img_set_name}_nuclei_{model_name}_segmentation_result"
            save_name_cell = f"{img_set_name}_cell_{model_name}_segmentation_result"
        if v2:
            nuc_masks = segment_nuclei_v3(
                rescaled_img,
                model,
                show_plot=save_plots,
                save_plots=save_plots,
                plot_dir=plot_dir,
                save_name=save_name_nuc,
            )
            save_masks(
                img_set_name,
                nuc_masks,
                outdir,
                image_ext=image_ext,
                mask_type="nuclei",
                model_name=model_name,
            )
            cell_masks = segment_cell_v3(
                rescaled_img,
                model,
                show_plot=save_plots,
                save_plots=save_plots,
                plot_dir=plot_dir,
                save_name=save_name_cell,
            )
            save_masks(
                img_set_name,
                cell_masks,
                outdir,
                image_ext=image_ext,
                mask_type="cell",
                model_name=model_name,
            )
        else:
            nuc_masks = segment_nuclei_v2(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name, nuc_masks, outdir, image_ext=image_ext, mask_type="nuclei"
            )
            cell_masks = segment_cell_v2(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name, cell_masks, outdir, image_ext=image_ext, mask_type="cell"
            )


def save_imageJ_masks(set_name, masks, outdir, image_ext=".tif", mask_type="cell"):
    """
    Save masks as ImageJ ROIs
    Parameters:
             set_name (str): the image set name to be saved in the output file
             masks (list of 2D or 3D arrays): the masks output from Cellpose
             outdir (Path or str): the output directory to save masks
             image_ext (str, optional): image extension, .tif by default
             mask_type (str, optional): specify the mask type to save, cell by default
    """
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    masks0 = io.imsave(outdir / (set_name + "_" + mask_type + "_masks" + masks_ext))
    io.save_rois(masks0, masks)


def preload_and_save_masks(
    ordered_files,
    outdir,
    masks_ext=".tif",
    mask_type="cell",
    nchannels=None,
    pretrained_model="cpsam_v2",
    gpu=True,
):
    """
    Load all images into memory and then batch-run cellpose on GPU
    ONLY use if image files are small, will crash with large files
    Parameters:
           ordered_files (list): a 1D list of file paths ordered by location,channel
           outdir (Path or str): the output directory to save masks
           image_ext (str, optional): image extension, .tif by default
           nchannels (int, optional): specify the number of channels, default = 4
    """
    model = load_model(pretrained_model=pretrained_model, gpu=gpu)
    if nchannels == None:  # handle default case when nchannels isn't specified
        nchannels = get_nchannels(ordered_files)

    grouped_files_by_channel = group_files_by_channel(ordered_files, nchannels)
    # if you have small images, you may want to load all of them first and then run, so that they can be batched together on the GPU
    print("loading images")
    imgs = load_image_set(
        [grouped_files_by_channel[i] for i in trange(len(grouped_files_by_channel))],
        nchannels,
    )

    print("running cellpose-SAM")
    flow_threshold = 0.4
    cellprob_threshold = 0
    tile_norm_blocksize = 0

    masks, flows, styles = model.eval(
        imgs,
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )

    print("saving masks")
    for i in trange(len(grouped_files_by_channel)):
        f = grouped_files_by_channel[i]
        set_name = get_image_set_name(f)
        io.imsave(outdir / (set_name + mask_type + "_masks" + masks_ext), masks[i])


def microfilm_plot(
    image_set,
    cmaps=[
        "pure_red",
        "pure_green",
        "pure_blue",
    ],  # use these if the cmap library isn't working
    # cmaps = ["gray_r", "cyan", "magenta"],
    channel_names=["MitoTracker", "LAMP1", "DAPI"],
    channel_label_show=True,
    scalebar_unit_per_pix=0.09,
    scalebar_size_in_units=3,
    fig_scaling=5.0,
    label_text="",
    ax=None,
    save=False,
    save_path=None,
):
    """Plots a microfigure using the microfilm library with specified parameters.
    See https://guiwitz.github.io/microfilm/notebooks/create_plots.html
    Args:
        image_set (list or np.array): A list of 2D arrays or a 3D array of images to be plotted, with shape (height, width, channels).
        cmaps (list): A list of colormaps for each channel. Default is ['pure_red', 'pure_green', 'pure_blue'].
        channel_names (list): A list of names for each channel. Default is ["MitoTracker", "LAMP1", "DAPI"].
        scalebar_unit_per_pix (float): The size of each pixel in micrometers. Default is 0.09.
        scalebar_size_in_units (float): The size of the scalebar in micrometers. Default is 3.
        fig_scaling (float): Scaling factor for the figure  size. Default is 5.0.
        label_text (str): Text to be added as a label on the figure. Default is an empty string.
        ax (matplotlib.axes.Axes): An optional matplotlib axes object to plot
        save (bool): Whether to save the figure. Default is False.
        save_path (str): The path to save the figure if save is True. Default is None.
    Returns:
        microfigure: The microfigure object created by the microfilm library.
    """
    import skimage.io
    from microfilm import microplot

    # Note that this library expects a CXY format for the images, so we need to swap axes from XYC to CXY
    if isinstance(image_set, list):
        image_set = np.stack(image_set, axis=-1)
    image_set = np.swapaxes(image_set, 0, -1)
    try:
        correct_cmap = image_set.shape[0] == len(cmaps)
        if correct_cmap or cmaps is None or image_set.ndim != 3:
            pass
        else:
            raise ValueError(
                f"Number of channels in image_set ({image_set.shape[0]}) does not match number of colormaps provided ({len(cmaps)})."
            )
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Using default colormaps instead.")
        cmaps = None
        channel_names = None

    # plt.figure(figsize=(10, 10))
    # don't show channel labels if an axis is provided or it will mess with formatting
    if ax:
        channel_label_show = False

    microfigure = microplot.microshow(
        images=image_set,
        cmaps=cmaps,
        flip_map=False,
        channel_label_show=channel_label_show,
        channel_names=channel_names,
        unit="um",
        scalebar_unit_per_pix=scalebar_unit_per_pix,
        scalebar_size_in_units=scalebar_size_in_units,
        scalebar_color="white",
        scalebar_font_size=12,
        fig_scaling=fig_scaling,
        ax=ax,
    )
    if label_text:
        microfigure.add_label(
            label_text=label_text, label_font_size=30, label_color="white"
        )
    if save and save_path is not None:
        microfigure.savefig(
            f"{save_path}/single.png", bbox_inches="tight", pad_inches=0, dpi=600
        )
    return microfigure
