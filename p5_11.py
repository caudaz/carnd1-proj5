#Generate images without having a window appear
import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import cv2
import glob
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from p5_lesson_functions import *
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, 
                        color_space='RGB', 
                        spatial_size=(32, 32),
                        hist_bins=32, 
                        orient=9, 
                        pix_per_cell=8, 
                        cell_per_block=2, 
                        hog_channel=0,
                        spatial_feat=True, 
                        hist_feat=True, 
                        hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
	

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
# based on DECISION FUNCTION
def search_windows2(img, 
                   windows, 
                   clf, 
                   scaler, 
                   color_space='RGB', 
                   spatial_size=(32, 32), 
                   hist_bins=32, 
                   hist_range=(0, 256), 
                   orient=9, 
                   pix_per_cell=8, 
                   cell_per_block=2, 
                   hog_channel=0, 
                   spatial_feat=True, 
                   hist_feat=True, 
                   hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        #prediction = clf.predict(test_features)
        dec_fun = clf.decision_function(test_features)
        prediction = int(dec_fun > 0.75)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

### TRAIN PARAMS -  Tweak and see how the results change.
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, None] # Min and max in y to search in slide_window()
    
TRAIN = True
if TRAIN:
    images_car    = glob.glob('vehicles/**/*.png', recursive=True)
    images_notcar = glob.glob('non-vehicles/**/*.png', recursive=True)    
    cars = []
    notcars = []    
    for image in images_car:
        cars.append(image)
    for image in images_notcar:
        notcars.append(image)
            
    car_features    = extract_features(cars, 
                                       color_space=color_space, 
                                       spatial_size=spatial_size, 
                                       hist_bins=hist_bins, 
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, 
                                       spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, 
                                       hog_feat=hog_feat)

    notcar_features = extract_features(notcars, 
                                       color_space=color_space, 
                                       spatial_size=spatial_size, 
                                       hist_bins=hist_bins, 
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, 
                                       spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, 
                                       hog_feat=hog_feat)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # now you can save it to a file
    with open('svc_classifier.pkl', 'wb') as f:
        pickle.dump(svc, f)
    with open('svc_classifier_X_scaler.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)

# and later you can load it
with open('svc_classifier.pkl', 'rb') as f:
    svc = pickle.load(f)
with open('svc_classifier_X_scaler.pkl', 'rb') as f:
    X_scaler = pickle.load(f)

# Create pre/post folders
video = 'project_video.mp4'
video_name, video_extension = os.path.splitext(video)
video_folder              = 'output_images'
video_folder_pre          = 'output_images/'+video_name + '_pre'
video_folder_post         = 'output_images/'+video_name + '_post'
video_folder_post2        = 'output_images/'+video_name + '_post2'
if not os.path.exists(video_folder): os.makedirs(video_folder)
if not os.path.exists(video_folder_pre): os.makedirs(video_folder_pre)
if not os.path.exists(video_folder_post): os.makedirs(video_folder_post)
if not os.path.exists(video_folder_post2): os.makedirs(video_folder_post2)

# Convert MP4 to JPGs
convert_mp4_to_jpgs = True
if convert_mp4_to_jpgs:
    vidcap = cv2.VideoCapture(video)
    fps    = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
      success, image = vidcap.read()
      if success:
         cv2.imwrite(video_folder_pre+'/frame{0:05}.jpg'.format(count), image)
      count += 1
    vidcap.release()


# Location of images to process
folder = video_folder_pre

# Heat thresold = 7 windows X 10 frames
heat_threshold = 7*10

list_peaks = []
list_areas = []
list_vols = []
list_cents = []

hot_wind1 = None
hot_wind2 = None
hot_wind3 = None
hot_wind4 = None
hot_wind5 = None
hot_wind6 = None
hot_wind7 = None
hot_wind8 = None
hot_wind9 = None
hot_wind10 = None

for file in os.listdir(folder):
    t=time.time()
    
    image = mpimg.imread(folder + '/' + file)
    print(folder + '/' +file)
    draw_image = np.copy(image)
    
    windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[380,520], xy_window=(96,96),   xy_overlap=(0.9, 0.9))
    windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[380,520], xy_window=(128,128), xy_overlap=(0.9, 0.9))
    windows4 = slide_window(image, x_start_stop=[None, None], y_start_stop=[380,620], xy_window=(160,160), xy_overlap=(0.8, 0.8))
    windows5 = slide_window(image, x_start_stop=[None, None], y_start_stop=[380,620], xy_window=(192,192), xy_overlap=(0.8, 0.8))
    
    windows = windows2 + windows3 + windows4 + windows5
    
    hot_windows = search_windows2(image, 
                                 windows, 
                                 svc, 
                                 X_scaler, 
                                 color_space=color_space, 
                                 spatial_size=spatial_size, 
                                 hist_bins=hist_bins, 
                                 orient=orient, 
                                 pix_per_cell=pix_per_cell, 
                                 cell_per_block=cell_per_block, 
                                 hog_channel=hog_channel, 
                                 spatial_feat=spatial_feat, 
                                 hist_feat=hist_feat, 
                                 hog_feat=hog_feat)                       


    # Add hot windows over 10 frames
    hot_wind1 = hot_wind2  if hot_wind2  is not None else []
    hot_wind2 = hot_wind3  if hot_wind3  is not None else []
    hot_wind3 = hot_wind4  if hot_wind4  is not None else []
    hot_wind4 = hot_wind5  if hot_wind5  is not None else []
    hot_wind5 = hot_wind6  if hot_wind6  is not None else []
    hot_wind6 = hot_wind7  if hot_wind7  is not None else []
    hot_wind7 = hot_wind8  if hot_wind8  is not None else []
    hot_wind8 = hot_wind9  if hot_wind9  is not None else []
    hot_wind9 = hot_wind10 if hot_wind10 is not None else []
    hot_wind10 = hot_windows
    hot_wind1thru5 = hot_wind1 + hot_wind2 + hot_wind3 + hot_wind4 + hot_wind5 + hot_wind6 + hot_wind7 + hot_wind8 + hot_wind9 + hot_wind10

    # Add heat to each box in box list
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_wind1thru5)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # Metrics for size of heat
    peak, area, vol, cent = labels_peak_area_vol_centr(heat, labels)
    list_peaks.append(peak)
    list_areas.append(area)
    list_vols.append(vol)
    list_cents.append(cent)

    t2 = time.time()    
    print(round(t2-t), 'Seconds to process 1 image')    


    # Process video 
    fig = plt.figure(figsize=(20.4, 10.0))

    plt.subplot(221)
    plt.title('HOT windows current frame')
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)    
    plt.imshow(window_img)

    plt.subplot(222)
    plt.title('HOT windows of 10 last frames')                        
    window_img = draw_boxes(draw_image, hot_wind1thru5, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)

    plt.subplot(223)
    plt.title('HEAT Windows After Threshold =' + str(heat_threshold) + '\n' + 
              'Peaks = ' + str(peak) + ' Areas=' + str(area) + ' Vols=' + str(vol) )
    plt.imshow(heatmap, cmap='hot')

    plt.subplot(224)
    plt.title('Cars found =' + str(labels[1]))
    # Plot heat after 1st threshold
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    plt.imshow(draw_img) 

    fig.tight_layout()
    plt.show()
    fig.savefig(video_folder_post2 + '/' + os.path.splitext(os.path.basename(file))[0] + '.jpg', dpi=fig.dpi)

    # Output video 
    plt.imshow(draw_img)
    mpimg.imsave(video_folder_post + '/' + os.path.splitext(os.path.basename(file))[0] + '.jpg', draw_img)   


np.save('list_peaks.npy',list_peaks)
np.save('list_areas.npy',list_areas)
np.save('list_vols.npy', list_vols)
np.save('list_cents.npy',list_cents)

fps=25
# Convert JPGs to MP4   
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
format = "XVID"
size = None
is_color=True
fourcc = VideoWriter_fourcc(*format)
vid = None

outvid = os.path.splitext(os.path.basename(video))[0]+'_output.mp4'
for file in glob.glob(video_folder_post+'/*.jpg'):
    img = imread(file)
    if vid is None:
        if size is None:
            size = img.shape[1], img.shape[0]
        vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    if size[0] != img.shape[1] and size[1] != img.shape[0]:
        img = resize(img, size)
    vid.write(img)
vid.release()

vid = None

outvid = os.path.splitext(os.path.basename(video))[0]+'_output2.mp4'
for file in glob.glob(video_folder_post2+'/*.jpg'):
    img = imread(file)
    if vid is None:
        if size is None:
            size = img.shape[1], img.shape[0]
        vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    if size[0] != img.shape[1] and size[1] != img.shape[0]:
        img = resize(img, size)
    vid.write(img)
vid.release()