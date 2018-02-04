
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/not_car.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/heatmap.png
[image9]: ./examples/result_1.png
[image10]: ./examples/result_2.png
[image11]: ./examples/result_3.png
[video1]: ./project_video.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook in the function called `get_hog_features`. I used the `hog` function from the `skimage` package.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and then after performing a grid search over the parameter spaces of the parameters for computing the features, I ended up using HOG features, spatially binned color and histograms of color with the following parameters:

```python
COLOR_SPACE = 'RGB' 
ORIENT = 9  
PIX_PER_CELL = 8 
CELL_PER_BLOCK = 2 
HOG_CHANNEL = "ALL"
SPATIAL_SIZE = (16, 16)
HIST_BINS = 16  

```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using the following parameters in the code cell right with the title `Create New Model` at the middle of the notebook. The library i used is `sklearn` . Using this classifier, I managed to get 0.9802 accuracy on the test set. 

```python
svc = LinearSVC(C=1.0, loss='hinge', max_iter=1000, random_state=0, verbose=1)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the `find_cars` function where i compute the HOG features just once for the entire image and then use a sliding-window mechanism over the results. In order to compensate for different sizes of the cars in the images I used multiple scales of the subimage I extracted and also multiple shapes of the subimage extracted. More specifically, the subimages which were extracted were both squared and rectangular (as in many frames, the cars were actually rectangular). Of course, after this step they were all resized to be of shape (64,64) as that was the size the training images had. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to speed up the performance I used multiple threads to compute both the features used for training and for computing the predictions on the subimages for the inference part. 

This can be easily observed in the functions `get_car_bboxes_multiproc` and `add_features_multiproc`. Here are some pictures to show the results:

 
 ![alt text][image9]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. This can be ovserved in the `create_img_mean_to_bbox_dict`

### Here are some frames and their corresponding heatmaps with the final bounding box drawn on the image

 ![alt text][image10]
 ![alt text][image11]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First of all, the main problem i encountered was the fact that the time needed to make predictions on all the subimages in a particular frame is quite substantial. This means that it's very difficult to analyze a vast amount of data in a reasonable ammount of time. 

Second of all, there were a lot of parameters to tune at various steps in the whole pipeline and finding the precise combination of parameteres which yield the optimal results is very time consuming as well.

One direction in which i would want to develop this further is to use a Deep Neural Network architecture which is specifically designed for object detection like SSD or YOLO in order to solve this task.
