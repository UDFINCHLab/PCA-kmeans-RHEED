import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import cv2 as cv
import h5py as h5
from os import makedirs

from typing import Optional, cast, Any



INPUT_FILE = ''
OUT_PATH = ''
OUT_NAME = ''
TITLE = ''

NUM_VECTORS = -1

FIG_SIZE = (7, 5)


def plot_pca(Input_File: str,
             Out_Path: str,
             Title: str='',
             Num_Vectors: int=-1,
             Fig_Size: tuple[float,float]=(7,5),
             show: bool=False) -> None:
    
    if Out_Path[-1] != '/':
        Out_Path += '/'

    plt.rc('font', family='Times New Roman', weight='bold', size=8)
    
    with h5.File(Input_File, 'r') as file:
        # Access the pca group
        pca_group: h5.Group = file['pca']
        
        # Load the eigenvectors file
        eigen_vectors = pca_group['eigenvectors'][:]
        
        # Load the eigenvalues file
        eigen_values = pca_group['eigenvalues'][:]
        
        # Access the original data group
        data_group: h5.Group = file['data']
        
        # Load the times file
        times = data_group['times'][:]
        
        width = data_group['image_data'].attrs['x_res']
        height = data_group['image_data'].attrs['y_res']
    
    eigen_vectors = np.transpose(eigen_vectors, (2, 1, 0))
    image_shape = (height, width)

    
    # Reshape the eigenvectors matrix
    eigen_vectors = reshape_eigen_vectors(eigen_vectors, image_shape)
    
    eigen_values, eigen_vectors = invert_eigens(eigen_values, eigen_vectors)
    
    eigen_vectors = scale_eigen_vectors(eigen_vectors, out_shape=(201,201)) 
    
    if Num_Vectors < eigen_vectors.shape[2] and Num_Vectors > 0:
        eigen_vectors = eigen_vectors[:,:,0:Num_Vectors]
        eigen_values = eigen_values[:,0:Num_Vectors]
    
    plot_eigenvalues_and_vectors(eigen_values, times, eigen_vectors, fig_size=Fig_Size, out_path=Out_Path, title=Title)
   
    return


def adjustments(fig_size: tuple[float,float])-> list[float]:
    
    #             [L,     B,     R,     T,     W,     H,     YT,     XT]
    if fig_size == (7, 2):
        adjusts = [0.06,  0.2,   0.99,  0.88,  0.2,   0.2,   -0.035, 0]
    elif fig_size == (3.5, 2):
        adjusts = [0.12,  0.2,   0.95,  0.88,  0.2,   0.2 ,  -0.065, 0]
    elif fig_size == (7, 5):
        adjusts = [0.075, 0.1,   1.03,  0.93,  0.2,   0.2,   -0.07, 0]
    elif fig_size == (7, 3.57):
        adjusts = [0.075, 0.12,  1.01,  0.87,  0.2,   0.2,   -0.07, 0]
    elif fig_size == (3.375, 3.375):
        adjusts = [0.11,  0.08,  0.97,  0.99,  0.0,   0.0,   -0.10, -0.16]
    elif fig_size == (3.375, 6.75):
        adjusts = [0.11,  0.05,  0.97,  0.99,  0.0,   0.0,   -0.15, -0.19]
    elif fig_size == (3.375, 4.5):
        adjusts = [0.11,  0.08,  0.97,  0.99,  0.0,   0.0,   -0.15, -0.19]
    else:
        adjusts = [0,     0,     1,     1,     0,     0,     0,     0]
        
    return adjusts


def scale_eigen_vectors(eigen_vectors: np.ndarray, out_shape: tuple[int,int]=(101,201)) -> np.ndarray:
    # Rescale the x and y dims of the eigen vectors to the output shape
    num_vectors = eigen_vectors.shape[2]
    new_vectors = np.empty((out_shape[0], out_shape[1], num_vectors))
    
    for i in range(num_vectors):
        image = cv.resize(eigen_vectors[:,:,i], out_shape[::-1], interpolation=cv.INTER_CUBIC)
        new_vectors[:,:,i] = image
        
    return new_vectors


def invert_eigens(eigen_values: np.ndarray, eigen_vectors: np.ndarray) -> np.ndarray:
    # If the eigenvector median is negative, invert the eigenvector and eigenvalue
    for i in range(eigen_values.shape[1]):
        mid_val = np.max(eigen_vectors[:,:,i]) - (np.max(eigen_vectors[:,:,i]) - np.min(eigen_vectors[:,:,i]))/2
        median = np.median(eigen_vectors[:,:,i])
        if median > mid_val:
            eigen_vectors[:,:,i] = -eigen_vectors[:,:,i]
            eigen_values[:,i] = -eigen_values[:,i]
    return  eigen_values, eigen_vectors


def reshape_eigen_vectors(eigen_vectors: np.ndarray, image_shape: list[int]) -> np.ndarray:
    
    num_vectors = eigen_vectors.shape[0]
    
    new_vectors = np.empty((image_shape[0], image_shape[1], num_vectors))
    
    for i in range(num_vectors):
        image = eigen_vectors[i,:].reshape([image_shape[1], -1]).T
        new_vectors[:,:,i] = image
    

    return new_vectors


def plot_eigenvalues_and_vectors(eigen_values: np.ndarray, times: np.ndarray, eigen_vectors: np.ndarray, fig_size: tuple[float,float], out_path: str, title: str='') -> None:
    
    number_of_eigenvalues = eigen_values.shape[1]
    
    gs = GridSpec(nrows=number_of_eigenvalues, ncols=2, width_ratios=[1, .85], hspace=-.02, wspace=0.01)
    
    fig = plt.figure(figsize=fig_size)
    
    adjusts = adjustments(fig_size)
    
    #Construct lists of axes for the left and right sides of the plot
    axes_left = np.empty(number_of_eigenvalues, dtype=object)
    axes_right = np.empty(number_of_eigenvalues, dtype=object)
    for i in range(number_of_eigenvalues):
        ax_left = fig.add_subplot(gs[i, 0])
        if i != 0 and i != number_of_eigenvalues-1:
            ax_left.set_xticklabels([])
        elif i == 0:
            ax_left.xaxis.tick_bottom()
            #ax_left.xaxis.tick_top()
            ax_left.set_xticklabels([])
            
        ax_right = fig.add_subplot(gs[i, 1])
        ax_right.axis('off')
        axes_left[i] = ax_left
        axes_right[i] = ax_right
        
    
        
    
        
    plot_all_eigenvalues(eigen_values, times, axes_left, fig_size)
    
    plot_all_eigen_vectors(eigen_vectors, axes_right)
    
    plt.tight_layout(pad=0)
    
    fig.suptitle(f'{title}', fontsize=12, fontweight='bold', x=0.5, y=adjusts[3] + 0.05)
    
    
    plt.subplots_adjust(left=adjusts[0], bottom=adjusts[1], right=adjusts[2], top=adjusts[3], wspace=adjusts[4], hspace=adjusts[5])
    
    plt.show()
    
    # Make the directory if it does not exist
    makedirs(out_path, exist_ok=True)
    
    fig.savefig(f'{out_path}Eigen_Values_and_Vectors.png', dpi=1000)

    return



def plot_all_eigen_vectors(eigen_vectors: np.ndarray, axes: Optional[np.ndarray]) -> None:
    
    if axes is None:
        number_of_eigenvectors = eigen_vectors.shape[-1]
        _, ax  = plt.subplots(nrows=number_of_eigenvectors, ncols=1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0})
    
        axes: list[plt.Axes] = ax.flatten()
        
    for adx, ax in enumerate(axes):
        ax: plt.Axes
        plot_eigen_vector(eigen_vectors[:,:, adx], ax=ax)
        
        
        
def plot_eigen_vector(eigen_vector: np.ndarray, ax: Optional[plt.Axes]) -> None:
    
    # Get the extreem value of the eigenvector
    extreem = np.max(np.abs(eigen_vector))
    
    # Scale the image to -1 to 0 to 1
    eigen_vector = eigen_vector / extreem
    
    # Ensure white corresponds to 0 using TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    # Plot the eigenvector
    plot = ax.imshow(eigen_vector, cmap=rwb_color_map(), interpolation='nearest', norm=norm)
    
    colorbar = plt.colorbar(plot, ax=ax, shrink=0.85)  # Adjust fraction for width

    # Customize colorbar
    colorbar.set_ticks([0, -1, 1])  
    colorbar.ax.tick_params(size=5)  
    colorbar.ax.tick_params(labelsize=8)
    
    
    ax.axis('off')
    
    return


def plot_all_eigenvalues(eigen_values: np.ndarray, times: np.ndarray, axes: Optional[np.ndarray], fig_size: Optional[tuple]) -> None:
    
    
    # Rescale times if they are too large
    times, unit = scale_times(times)
    
    if axes is None:
        number_of_eigenvalues = eigen_values.shape[1]
        fig, ax  = plt.subplots(nrows=number_of_eigenvalues, ncols=1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0})
    
        axes: list[plt.Axes] = ax.flatten()
        
        top_level = True
    
    else:
        top_level = False
        
    adjusts = adjustments(fig_size)
    
    for adx, ax in enumerate(axes):
        ax: plt.Axes
        plot_eigenvalues(eigen_values[:,adx], times, ax=ax)
        ax.set_ylabel(f'Component {adx+1}', fontsize=8, fontweight='bold')
        ax.yaxis.set_label_coords(adjusts[6], 0.5)
        
    # Set the x-axis label
    axes[-1].set_xlabel(f'Time ({unit})', fontsize=8, fontweight='bold')
    axes[-1].xaxis.set_label_coords(0.5, adjusts[7])



    if top_level:
        
        # Set the figure title
        fig.suptitle(f'{TITLE} Eigenvalues', fontsize=12, fontweight='bold')
        
        plt.show()
        
        # Save the plot at a specific resolution
        fig.savefig(f'{OUT_PATH}{OUT_NAME}Eigen_Values.png', dpi=1000)
    
    return



def plot_eigenvalues(eigenvalues: np.ndarray, times: np.ndarray, ax: Optional[plt.Axes]=None) -> None:
    
    if ax is None:
        _, ax  = plt.subplots(1, 1, figsize=(7, 1.5))
        
    ax.scatter(times, eigenvalues, label='', color='black', linewidth=1, marker='o', s=1, zorder=4)
    
    #  ax.set_facecolor(background_color())
    
    ax.grid(True)
    
    # Select the tick mark that is farthest from zero
    max_tick = np.ceil(np.max(np.abs(eigenvalues)))
    
    # Set the new tick marks to be the same as the +-max tick and zero
    ax.set_yticks([-max_tick, 0, max_tick])
    
    # Set the axis limits to be 15% larger than the max tick
    ax.set_ylim(-1.2*max_tick, 1.2*max_tick)

    ax.set_xlim(times[0], times[-1])
    
    return
    


def scale_times(times: np.ndarray) -> tuple[np.ndarray, str]:
    
    max_val = times.max()
    
    if max_val > 60:
        new_times = times / 60
        max_val = new_times.max()
        unit = 'minutes'
        
        if max_val > 60:
            new_times = new_times / 60
            unit = 'hours'
            
    else:
        new_times = times
        unit = 'seconds'

    return new_times, unit



def au_color_map() -> LinearSegmentedColormap:
    # Define the RGB values for the start and end colors
    start_color = (11/255, 35/255, 65/255)  # Blue
    end_color = (242/255, 107/255, 5/255)    # Orange

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color], N=256)
    
    return cmap


def rwb_color_map() -> LinearSegmentedColormap:
    # Define the RGB values for the start and end colors
    start_color = (0, 0, 1)  # Blue
    middle_color = (1, 1, 1) # White
    end_color = (1, 0, 0)    # Red

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=256)
    
    return cmap


def auw_color_map() -> LinearSegmentedColormap:
    # Define the RGB values for the start and end colors
    start_color = (11/255, 35/255, 65/255)  # Blue
    middle_color = (1, 1, 1) # White
    end_color = (242/255, 107/255, 5/255)    # Orange

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=256)
    
    return cmap

def train_color() -> str:
    return '#0b2341'

def val_color() -> str:
    return '#e86100'

def test_color() -> str:
    return '#215834'

def background_color() -> str:
    return '#ced3d9'

















if __name__ == '__main__': 
    plot_pca(INPUT_FILE, OUT_PATH, OUT_NAME, TITLE, NUM_VECTORS, FIG_SIZE)