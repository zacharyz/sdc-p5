**Vehicle Detection**

* Gather and organize data
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Train a Linear SVM classifier on normalized HOG features
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the above pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4

[viz0]: ./output_images/vis-car_not_car.png
[viz1]: ./output_images/vis-detectcars-HLS.png
[viz2]: ./output_images/vis-detectcars-HSV.png
[viz3]: ./output_images/vis-detectcars-RGB.png
[viz4]: ./output_images/vis-detectcars-YCrCb.png

[viztest]: ./output_images/vis-test-images.png

[vizhog1]: ./output_images/vis-hog-HLS-Car.png
[vizhog2]: ./output_images/vis-hog-HLS-Not-Car.png
[vizhog3]: ./output_images/vis-hog-HSV-Car.png
[vizhog4]: ./output_images/vis-hog-HSV-Not-Car.png
[vizhog5]: ./output_images/vis-hog-LUV-Car.png
[vizhog6]: ./output_images/vis-hog-LUV-Not-Car.png
[vizhog7]: ./output_images/vis-hog-RGB-Car.png
[vizhog8]: ./output_images/vis-hog-RGB-Not-Car.png
[vizhog9]: ./output_images/vis-hog-YCrCb-Car.png
[vizhog10]: ./output_images/vis-hog-YCrCb-Not-Car.png


[heat1]: ./output_images/vis-frame-7.png
[heat2]: ./output_images/vis-frame-11.png
[heat3]: ./output_images/vis-frame-21.png
[heat4]: ./output_images/vis-frame-28.png
[heat5]: ./output_images/vis-frame-32.png
[heat6]: ./output_images/vis-frame-35.png

[lab1]: ./output_images/vis-frame-7-labels.png
[lab2]: ./output_images/vis-frame-11-labels.png
[lab3]: ./output_images/vis-frame-21-labels.png
[lab4]: ./output_images/vis-frame-28-labels.png
[lab5]: ./output_images/vis-frame-32-labels.png
[lab6]: ./output_images/vis-frame-35-labels.png


[resdet]: ./output_images/vis-frame-35-detected.png



## Pipeline Details

The code for this project can be found in the jupyter notebook titled `Vehicle-Detection`. It is split up into three primary sections - `helper functions`, `Pipeline` and `Visualize`. The `Pipeline` class includes functions for training the classifier, detecting cars and processing individual frames of the video. The `Visualize` class includes functions that aided in the generation of images for this writeup.

### Gather and Organize Data

I downloaded and used the data provided by Udacity for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images that were sized 64x64 pixels. The images were extracted from the [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets.

These images were initially loaded into the `Pipeline` at the start of the `train_classifier` function using `get_image_files`, which gets all the path names for `car` and `notcar` images in the dataset.

Here is a random example of a `car` and `notcar`
![examplecar][viz0]

Total data I had for training:
```
Number of car images: 8792
Number of non-car images: 8968
```


### Histogram of Oriented Gradients (HOG)

The following code found in `train_classifier` is where I use my lists of `cars` and `notcars` to extract HOG features.


```
car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
```         

The above use of `extract_features` uses the `get_hog_features` helper function which uses scikit-learn's `hog()` function to extract HOG features.

I settled on the following HOG parameters:
```
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

Here are some examples generated by the `Visualize` class's `visualize_hog` function for each colorspace and the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][vizhog1]
![alt text][vizhog2]
![alt text][vizhog3]
![alt text][vizhog4]
![alt text][vizhog5]
![alt text][vizhog6]
![alt text][vizhog7]
![alt text][vizhog8]
![alt text][vizhog9]
![alt text][vizhog10]

### Train a linear SVM classfier

The code for training the linear SVM can be found in `Pipeline`'s `train_classifier` function.

I randomized and split my data into 80% training and 20% test data using sklearn's `train_test_split`.

I normalized the features by using scikit-learn's StandardScaler.

For each of the color spaces I then trained my classifier with the following results:

```
Extracting features for color_space RGB
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
19.76 Seconds to train SVC...
Test Accuracy of SVC =  0.9842
Extracting features for color_space HSV
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
14.42 Seconds to train SVC...
Test Accuracy of SVC =  0.9887
Extracting features for color_space HLS
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
14.71 Seconds to train SVC...
Test Accuracy of SVC =  0.9927
Extracting features for color_space YCrCb
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
13.59 Seconds to train SVC...
Test Accuracy of SVC =  0.9935
```

Of note is the `YCrCb` color space's test accuracy of 99.35%

I stored the result of each color space training into seperate pickle files for later experimentation.

### Sliding Window Search

It is more efficient to extract the HOG features for the entire region of interest. This way we only perform the HOG extraction once and not for every window in our sliding window search. I limited my window search between the vertical region of starting at ystart = 400 and ending at ystop = 656 with best results at a window scale = 1.5.

I found at scale of 1.0 that enough cars were detected. I am on a decently beefy system so it was easy to test for 4 scales.

Here are example detections for a few different color spaces.

![alt text][viz1]
![alt text][viz2]
![alt text][viz3]
![alt text][viz4]

Ultimately I searched on 4 scales [1.0, 1.5, 1.75, 2.0], using ALL channels of YCrCb HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][viztest]
---

### Video Implementation

Here's a [link to my video result](./project_video_output.mp4), output of the test video can be found [here](./test_video_output3.mp4)

To achieve these results I processed each frame using `Pipeline`'s `process_image`. 

The first thing I did was to get all teh bounding boxes for all the detections. Then used the trained classifier to detect the same car every time. But since I use 4 different scales thre will be multiple bounding boxes within or around each car. 

In order to filter out false detections I created a heatmap and applied an initial threshold of using `apply_threshold`.I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.   

I also saved each heatmap and once 30 were detected I stacked them and applied a threshold of 5 to give me more stable bounding boxes. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]
![alt text][heat5]
![alt text][heat6]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][lab1]
![alt text][lab2]
![alt text][lab3]
![alt text][lab4]
![alt text][lab5]
![alt text][lab6]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][resdet]



---

### Discussion

One of the main challenges for this project were determining what color space to use and tuning the rest of the parameters to reduce false positives. I was able to use the classifiers final reported accuracy and visually guage that of my options YRcCB produced the best results.

This particular pipeline is limited by its region of interest which is heavily dependent on a flat/straight road. A curved or pitched road will drastically change its effectiveness. I tried to account for false positives that were present in the videos but not ones that may exist in other situations not featured in the test videos.

Finally this isn't a real time detector. A full project video takes me about 25 minutes to process. I hope to experiment with faster deep-learning based approaches in the future such as [SSD](https://arxiv.org/abs/1512.02325) or [YOLO](https://arxiv.org/abs/1506.02640), which can achieve real-time speeds of 30+ fps.
