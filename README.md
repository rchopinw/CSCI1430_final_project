Brown University CSCI1430 Computer Vision Final Project: Real-time Face Mask Detection System

Contributors: Bangxi Xiao, Qiubai Yu, Xinyu Song, Tianqi Liu

Contact: bangxi_xiao@brown.edu / qiubai_yu@brown.edu / xinyu_song@brown.edu / tianqi_liu@brown.edu

Usage (preparation):

1. Cloning the git repository to local. 
2. Open train_log and you will find a bunch of model zip files.
3. Unzip the one ending with ".zip" to directory "model".
4. The desired file structure of "train_logs" is: train_output/log - where the training log data are stored; train_output/model - where the trained models are stored; train_output/evaluate.txt - the evaluation (test) results of all the models.

Usage (training, validation, and testing tfrecords data generation):

1. The original dataset for training and validation can be found via: https://www.kaggle.com/datasets/tapakah68/medical-masks-part1
2. The original testing data can be found via: https://www.kaggle.com/datasets/tapakah68/medical-masks-part7
3. After downloading and unzipping the files, please create another file 
