Brown University CSCI1430 Computer Vision Final Project: Real-time Face Mask Detection System

Contributors: Bangxi Xiao, Qiubai Yu, Xinyu Song, Tianqi Liu

Contact: bangxi_xiao@brown.edu / qiubai_yu@brown.edu / xinyu_song@brown.edu / tianqi_liu@brown.edu

FIRST THING FIRST: Install the required packages from requirements.txt

Usage (model preparation):

1. Cloning the git repository to local. 
2. Open train_log and you will find a bunch of model zip files.
3. Unzip the one ending with ".zip" to directory "model".
4. The desired file structure of "train_logs" is: train_output/log - where the training log data are stored; train_output/model - where the trained models are stored; train_output/evaluate.txt - the evaluation (test) results of all the models.

Usage (training, validation, and testing tfrecords data generation):

1. The original dataset for training and validation can be found via: https://www.kaggle.com/datasets/tapakah68/medical-masks-part1
2. The original testing data can be found via: https://www.kaggle.com/datasets/tapakah68/medical-masks-part7
3. After downloading and unzipping the files, place the images from 1 into data/train_images and images from 2 into data/test_images (according to ProcessARGS.py).
4. Run script convert_data.py and the tfrecords files will be automatically written into data/train_validation_data and data/test_data.
5. (Alternative) If you don't want to start from raw images, feel free to download the prepared tfrecords from our public GCP bucket: gs://csci1430-final/data

Usage (model training):

1. After data preparation, you can train the models by running script "train_model.py".
2. To train the model, you can use command line "python train_model.py --help" to learn more about the available options. 
3. (Alternative) You can also fetch the models directly from our GCP bucket: gs://csci1430-final/train_output/model
4. For the training logs, they will be written into train_output/log, separated by different models. (You can also obtain the training logs from the bucket: gs://csci1430-final/train_output/log)

Usage (detection system):
1. Make sure that your PC has a camera.
2. Running script "run_detection.py". You might also want to learn more about the options we provide via "python run_detection.py --help".

Other notes:
To download data from GCP bucket, gsutil is needed. 
