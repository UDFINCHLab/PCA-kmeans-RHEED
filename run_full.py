# Imports
from pre_processing import pre_processing
from pca import pca
from k_means import k_means
from plot_pca import plot_pca
from plot_kmeans import plot_k_means


####################### Input/Output Settings #######################
IN_FILE = ''
DATA_OUT_PATH = '' 
DATA_OUT_NAME = '' # Will be an h5 file 
IMAGE_OUT_PATH = ''
####################### Preprocessing Settings #######################


FRAME_PERIOD = 1
BLANK_THRESH = 10

KILL_BR = False

SCAR_REMOVAL = False
SCAR_SCOPE = (20, 20, 150, 25) # (top, bottom, left, right)
PASSES = 10

CROP_EDGES = False
CROP = [175, 315, 200, 460] # [top, bottom, left, right]

FAKE_HDR = False
HDR_TYPE = 'Gamma' # 'CLAHE' or 'Gamma' or 'Gamma2' or 'Invert'
HDR_RESCALE = False
GAMMA = 2
SIGMA = 15

BKGRND_CROP = False
BKGRND_RESCALE = False
BKGRND_EACH = False
BKGRND_THRESH = 50

TRANSLATE = False
TSHIFT = (100, 100) # (x, y)

DRIFT = False
D_AMOUNT = (25, 25) # (x, y)
D_TYPE = 'Linear' # 'Linear' or 'Jumpy'
JUMP_SIZE = (5, 1) # (x, y)

RSS = False
FIRST_ONLY = False
VERTICAL = False
RSS_SCOPE = (10, 10) # (left, right)
RSS_CROP = [20, 20, 20, 20] # [top, bottom, left, right] all positive


INVERT = False

####################### PCA Settings #######################
PCA_COMPONENTS = 6

####################### k-means Settings #######################
RUNS = 100
CLUSTERS = (4, 4) # Can be a single value, a tuple, or a list of values

####################### PCA Plot Settings #######################
PCA_FIG_SIZE = (3.375, 6.75)
NUM_VECTORS = -1 # -1 for all, otherwise a positive integer

####################### k-means Plot Settings #######################
MEANS_FIG_SIZE = (3.375, 3.375)

####################### Call Subrutines #######################

print("Plots will be saved to: ", IMAGE_OUT_PATH)

try:
    pre_processing_location = pre_processing(IN_FILE, DATA_OUT_PATH, DATA_OUT_NAME, FRAME_PERIOD, BLANK_THRESH, 
                    KILL_BR, SCAR_REMOVAL, SCAR_SCOPE, PASSES, CROP_EDGES, 
                    CROP, FAKE_HDR, HDR_TYPE, HDR_RESCALE, GAMMA, SIGMA, 
                    BKGRND_CROP, BKGRND_RESCALE, BKGRND_EACH, BKGRND_THRESH, 
                    TRANSLATE, TSHIFT, DRIFT, D_AMOUNT, D_TYPE, JUMP_SIZE, RSS, 
                    FIRST_ONLY, VERTICAL, RSS_SCOPE, RSS_CROP
                    )

except Exception as e:
    raise Exception('Pre-processing failed') from e


pca_result = pca(Input_File=pre_processing_location, PCA_Components=PCA_COMPONENTS)


if pca_result is False:
    raise Exception('PCA failed')
else:
    print('PCA Complete')
    
    
kmeans_result = k_means(Input_File=pre_processing_location, Runs=RUNS, Clusters=CLUSTERS)


if kmeans_result is False:
    raise Exception('k-means failed')
else:
    print('k-means Complete')
    
plot_k_means(Input_File=pre_processing_location, Out_Path=IMAGE_OUT_PATH, Fig_size=MEANS_FIG_SIZE, show=True)

print('k-means Plotting Complete')

plot_pca(Input_File=pre_processing_location, Out_Path=IMAGE_OUT_PATH, Fig_Size=PCA_FIG_SIZE, Num_Vectors=NUM_VECTORS, show=True)

print('PCA Plotting Complete')
    
    


    
    



