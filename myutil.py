'''
Overwriting small modifications on functions directly taken from neurite package.
Citation: Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import glob
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting


def slices(slices_in,           # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,          # option to plot the images in a grid or a single row
           width=15,            # width in in
           show=False,           # option to actually show the plot (plt.show())
           savepath=None,       # path to save the figure
           loss_label=None,
           axes_off=True,
           plot_block=True,     # option to plt.show()
           facecolor=None,
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    slices_in = list(map(np.squeeze, slices_in))
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, \
                'each slice has to be 2d or RGB (3 channels)'

    def input_check(inputs, nb_plots, name, default=None):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [default]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps', default='gray')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])
        
        # MODIFIED LABEL #########################################
        if i == 0 and loss_label is not None:
            ax.set_ylabel(loss_label, fontsize=12)

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i],
                          interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:  # and cmaps[i] is not None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if facecolor is not None:
        fig.set_facecolor(facecolor)

    if show:
        plt.tight_layout()
        plt.show(block=plot_block)
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)

    return (fig, axs)



def flow(slices_in,           # the 2D slices
         titles=None,         # list of titles
         cmaps=None,          # list of colormaps
         width=15,            # width in in
         indexing='ij',       # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,   # whether to match the image view, i.e. flip y axis
         grid=False,          # option to plot the images in a grid or a single row
         show=False,           # option to actually show the plot (plt.show())
         savepath=None,       # path to save the figure
         quiver_width=None,
         plot_block=True,  # option to plt.show()
         scale=1):            # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    slices_in = np.copy(slices_in)  # Since img_indexing, indexing may modify slices_in in memory

    if indexing == 'ij':
        for si, slc in enumerate(slices_in):
            # Make y values negative so y-axis will point down in plot
            slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show(block=plot_block)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)

    return (fig, axs)

def get_oasis_data(path):
    """
    Given the OASIS dataset at PATH, get all the 2D slice images and load them as a single torch tensor.
    """

    # Get all slice file paths
    names = sorted(glob.glob(path + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))[0:255]

    # Load all slices into a list of tensors
    slices = []
    for file_path in names:
        img = nib.load(file_path).get_fdata()
        img_tensor = torch.tensor(img, dtype=torch.float32)
        slices.append(img_tensor)

    # Stack all slices into a single tensor
    all_slices = torch.stack(slices).permute(0, 3, 1, 2)  # (H, W, B) -> (B, H, W)
    assert all_slices.dim() == 4
    assert all_slices.shape[1] == 1

    return all_slices


def make_slices_fig(path, columns=1):
    '''
    Take all the example_slices*.png files in the path, stack them vertically into a single image,
    and add text labels on the left side for each image. Pad with white if the widths of the images differ.
    '''

    file_paths = sorted(glob.glob(path + '/example_slices*.png'))
    images = [plt.imread(file_path) for file_path in file_paths]

    # Convert all images to grayscale
    images = [img if img.ndim == 2 else img[:, :, 0] for img in images]

    # Extract labels from filenames
    labels = [file_path.split('_')[-1].split('.png')[0] for file_path in file_paths]

    num_images = len(images)
    rows_per_column = (num_images + columns - 1) // columns

    # Define a fixed padding width
    font_size = 45
    label_width = 150  # Fixed padding width, ensure it's sufficient for all labels

    # Determine the maximum width of all images
    max_width = max(img.shape[1] for img in images)

    stacks = []
    for col in range(columns):
        start_idx = col * rows_per_column
        end_idx = min(start_idx + rows_per_column, num_images)
        if start_idx < num_images:
            # Add labels to the left of each image
            labeled_images = []
            for img, label in zip(images[start_idx:end_idx], labels[start_idx:end_idx]):
                # Pad the image to the maximum width with white
                if img.shape[1] < max_width:
                    padding = max_width - img.shape[1]
                    img = np.pad(img, ((0, 0), (0, padding)), mode='constant', constant_values=1)
                    print(f"Padding image {label} from width {img.shape[1]} to width {max_width}")

                # Create a label image
                label_img = np.ones((img.shape[0], label_width))  # Create a white image for the label
                fig, ax = plt.subplots(figsize=(label_width / 100, img.shape[0] / 100), dpi=100)
                ax.text(0.5, 0.5, label, fontsize=font_size, ha='center', va='center', rotation=90)
                ax.axis('off')
                fig.canvas.draw()
                label_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, 0] / 255.0  # Extract grayscale
                plt.close(fig)

                # Combine the label image and the original image
                labeled_img = np.hstack((label_img, img))
                labeled_images.append(labeled_img)

            stack = np.vstack(labeled_images)
            stacks.append(stack)

    final_image = np.hstack(stacks)

    save_path = path + '/combined_slices.png'
    # Normalize final_image to ensure values are in the range [0, 1]
    final_image = final_image / 255.0 if final_image.max() > 1 else final_image
    plt.imsave(save_path, final_image, cmap='gray')

    return save_path

