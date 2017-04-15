---

# Behavioral Cloning

[image1]: ./images/track1_normal_center.jpg "Track 1 Center Normal"
[image2]: ./images/track1_reverse_center.jpg "Track 1 Center Reverse"
[image3]: ./images/track1_recover_left1.jpg "Track 1 Recover Left 1"
[image4]: ./images/track1_recover_left2.jpg "Track 1 Recover Left 2"
[image5]: ./images/track1_recover_left3.jpg "Track 1 Recover Left 3"
[image3]: ./images/track1_recover_right1.jpg "Track 1 Recover Right 1"
[image4]: ./images/track1_recover_right2.jpg "Track 1 Recover Right 2"
[image5]: ./images/track1_recover_right3.jpg "Track 1 Recover Right 3"

---
### Files Submitted

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

A model can be trained by running the model.py file and providing it with training data and an output file
```sh
python model.py "data/track1_normal data/track1_reverse ..." --output_file model.h5
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

My model mimics the NVIDIA self driving car convolutional neural network as described in [^1] but with a few tweaks to fit the problem space of this project. At first I started with a very simple architecture that just ran the input images through 2 fully connected layers and outputted a steering angle. However, this model ended up being too simple and had high bias. So I decided to research some known architectures that have performed well on this problem. I read about the NVIDIA architecture and tested it with immediate success. It lowered the bias of the network significantly and my car was driving much better.

1. First I crop the image to remove the portion of the image above the horizon and remove the bottom of the image which includes the car hood. This cropping allows the network to focus on the important features on the road and removing the possibility of making false correlations with features not on the road.

2. Next I normalize the input images to have values between (-.5, .5). This will allow the distrubtion of values to be centered around 0 and allow the relu activation functions to perform better.

3. Then I run the normalized input data through 4 convolutional layers of increasing filter sizes each followed by a relu activation function. First are 2 layers with kernel size 5x5 and 2x2 stride and output filters of 24 and 36 respectively. The next 2 layers have kernel size of 3x3 and 1x1 strides with output filters of size 48 and 64 respectively. These convolutional layers allow the network to perform feature extraction from the input images.

4. The feature extraction outputs are then flattened and passed through 4 fully connected layers of output sizes 100, 50, 10, 1. These layers serve to convert the extracted features into a corresponding steering angle.

### Final Model Architecture
- Crop 	- Top, Bottom
- Lambda 	- x / 255 - 0.5 (output (-0.5, 0.5))
- Conv2D 	- kernel (5x5), stride (2x2), filters (24)
- Conv2D 	- kernel (5x5), stride (2x2), filters (36)
- Conv2D 	- kernel (3x3), stride (1x1), filters (48)
- Conv2D 	- kernel (3x3), stride (1x1), filters (64)
- Flatten
- Dense	- output (100)
- Dense	- output (50)
- Dense	- output (10)
- Dense	- output (1)

### Data Collection

The first time around I collected data by recording a few forward laps and a few reverse laps. The car performed ok but had trouble staying on the road at sharp turns. To combat this I recorded a few laps of the car driving from the side of the road back to the center. Now the car could handle the sharp turns but tended to drive off the road at section where the lines markings were missing. So I just recorded driving past that section of the road. Now the car could handle that specific situation but now it had trouble crossing the bridge. I realized that I was making the car memorize the road and therefore overfitting the specific sections of the road. The network was not becoming more robust to handle any situation. I wanted to remove my last set of training data, however, all of the data was lumped into a single directory. So, I reorganized my training data into separate directories for each type of training. I ended up with the following training directories:
#### track1_normal
![alt text][image1]

#### track1_reverse
![alt text][image2]

#### track1_realign_left
![alt text][image3]
![alt text][image4]
![alt text][image5]

#### track1_realign_right
![alt text][image6]
![alt text][image7]
![alt text][image8]

Now that I had my data split I could retrain my model using different combinations of training data. If any training strategy seemed to make the car perform worse I could easily remove that data from the process. 

### Training Strategy

During training I provide a list of directories that contain the data I want to train my model on. After collection I had about 32K images. In addition to the provided data I also augment the images by flipping them and inverting the output steering angle. This doubles the amount of data I have to train on giving me about 64K images. The loadData function will combine all of the data and split it into training/validation sets with an 80/20% split. 

During training the network is provided with training/validation data generators that generate the input data on the fly rather than loading all of the images in memory at once. The generator will first randomly shuffle all of the input data. Then it splits the data into small batches (32 images per batch). The model uses the center, left and right camera images for training data. For each left image I add a correction angle of .25 to teach the car to move from the left side of the road back to the center. For each right image I subtract a correction angle of .25 to teach the car to move from the right side of the road back to the center. Finally then generator will run a preprocessing step on the images. I first attempted to grayscale the images to see if that would help overfitting. However, that seemed to make the network perform worse. Currently, the preprocessing steps just returns the input image. I left the preprocessing function in the pipeline in case I wanted to add other processing later.

#### Attempts to reduce overfitting in the model 

My first attempt to reduce overfitting was to simple reduce that amount of situational specific training data. That worked surprisingly well to the point that the car could now drive all the way around the track without going offroad.
Also the training and validation loss was quite low, so I didn't need to add any regularization layers to the network. For future improvements I would add regularization once I began training for other tracks.

### Model parameter tuning

- learning rate - The model used an adam optimizer, so the learning rate was not tuned manually. 
- batch sizes   - I experimented with a batch sizes of 32, 64, and 128 and 32 performed the best, so I stuck with that.
- epochs        - I determined the best number of epochs by plotting the learning curves and checking at which epoch the model began to introduce higher variance between the training and validation curves. I determined that 5 epochs seemed to perform the best.
- left/right image correction angle - I started with a small correction angle of .1 and it didn't seem to contribute much to correcting the car when it deviated from the center of the road. Next I went with another extreme of .4 correction angle. This caused the car drive very erractically since it was constantly correcting itself. I finally settled with a correction angle of .25 which seemed to allow the car to recover from almost off road situations (like sharp turns) and it didn't cause the car to drive erratically. 

### Track 2 performance

I ran my model on track 2 using only the training from the first track and I was suprised how well it actually performed given the difficulty of the road. It did end up crashing in a section that had a sharp right curve followed by a sharp left curve, but I was impressed that it had made it that far. I also noticed that it had no concept of staying on the right side of the road. That is because the first track did not have a center line and the car was not train with the knowledge of a center line.

### Future improvements

1. Handle dynamic horizon lines introduced by elevation changes. When going downhill the horizon moves towards the bottom of the image so most of the image is above the horizon. In contrast when going uphill the horizon moves up the image so most of the image is below the horizon.

2. Train the car on track 2. The car needs to be trained to learn that it needs to stay on the right side of the road. Track 2 also, introduces a lot of other challenges such as switch backs, cliffs, and new road signs/ markers.

3. Render lane lines on the input images to provide more obvious features for the network to learn from.

4. Train/Predict throttle. This would probably require a new set of training data that was focused on throttle control. For my original training I mostly used full throttle except on very sharp turns. So if I used that data for training throttle, it would almost always go full speed.

5. Add regularization layers. For training only on the first simple track regularization wasn't needed to get a successful run. However, when I train the the second track and other potential tracks, regularization would be needed in order to generalize.
 
# References
[^1]: arXiv:1604.07316v1 [https://arxiv.org/abs/1604.07316]
