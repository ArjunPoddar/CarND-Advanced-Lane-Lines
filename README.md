## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is, given a video of a road taken from a car moving forward, to identify the lane lines and plot them on the video.

Here is a glimpse of the what we have accomplished in this project -

<p float="left">
    <img src='data/test_videos/gif_challenge_sample.gif' width="360" height="200"/>
    <img src ='data/output_videos/gif_solution_sample.gif' width="360" height="200"/>
</p>

&nbsp;

The work is done in the following jupyter notebook: [advanced_lane_lines.ipynb](advanced_lane_lines.ipynb)

The results of applying the image processing pipeline on the test images are here: [data/output_images/](data/output_images/)

The final result(plotted lanes on the project-video) can be found here: [data/output_videos/challenge_video_solution.mp4](data/output_videos/project_video_solution.mp4)



* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration matrix and the distortion correction coefficients are in the **Camera Calibration** section of the [advanced_lane_lines.ipynb](advanced_lane_lines.ipynb) Jupyter Notebook.

The goal is this part is to use known images to device a mapping function between distorted and undistorted images that when applied to other distorted images corrects the distortion. Images of chessboards are ideal candidates for this task as layout of chessboards is very mathematically structured and hence the corners on undistorted images are deterministic to some degree.

I read the given 20 distorted images of chessboards. Using the ```cv2.findChessboardCorners ``` function I tried to locate the corners of the chessboard and then using the ```cv2.drawChessboardCorners``` function I plotted the corners on the respective images. Corners were found for most of the images and for some of them, I could not find any corners such as for calibration4.jpg, calibration5.jpg etc. For the images where corners were found, the image-points(corners) were stored in an array named ```imgpoint``` which is an 2D array of image points as the given images are in 2D. The known corners of the undistorted chessboards, called the object points, are stored in a 3D array named ```objpoint```. We need 3D array for object points as we are trying to correct for distortion of 3D objects projected into 2D images. As chessboards the flat, the 3rd co-ordinate in ```objpoint``` is always kept at 0.

After that, I used the ```cv2.calibrateCamera``` function on the image points and the object points to calculate the camera matrix and the distortion coefficients, which are then used to get the undistorted image using the ```cv2.undistort``` function.

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
dst = cv2.undistort(image, mtx, dist, None, mtx)
```
Here is the result of undistorting one chessboard image as test:

![chess_undistort](data/pipeline_examples/undistorted_chessboard.jpg)

I have saved the camera calibration results [here](data/camera_calibration.p) for later uses. 


### Pipeline (single images)

#### 1. Distortion Correction

Using the camera calibration matrix and distortion coefficients that I had saved in the previous step, I write a function called ```undistort_image``` that, given a distorted image, performs undistortion and returns a corrected image. Here is an example: 

![image_undistort](data/pipeline_examples/step1_undistorted.jpg)

#### 2. Thresholding into a Binary Image: Color transforms, Gradients or other methods



<img src="data/pipeline_examples/step2_thresholding.jpg" width="800" height="500"/>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:


### Pipeline on all test images

### result: test1    
<img src="data/output_images/pipeline_result_test1.jpg" width="600" height="400">

&nbsp;
### result: test2
<img src="data/output_images/pipeline_result_test2.jpg" width="600" height="400">

&nbsp;
### result: test3
<img src="data/output_images/pipeline_result_test3.jpg" width="600" height="400">

&nbsp;
### result: test4
<img src="data/output_images/pipeline_result_test4.jpg" width="600" height="400">

&nbsp;
### result: test5
<img src="data/output_images/pipeline_result_test5.jpg" width="600" height="400">

&nbsp;
### result: test6
<img src="data/output_images/pipeline_result_test6.jpg" width="600" height="400">

&nbsp;
### result: straight_lines1
<img src="data/output_images/pipeline_result_straight_lines1.jpg" width="600" height="400">

&nbsp;
### result: straight_lines2
<img src="data/output_images/pipeline_result_straight_lines2.jpg" width="600" height="400">


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

## FInal Result 
### ( 0:00 - 0:25 sec )
![](data/output_videos/gif_solution_1.gif)

### ( 0:25 - 0:50 sec )
![](data/output_videos/gif_solution_2.gif)

Here's a [link to my video result](data/output_videos/project_video_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 
