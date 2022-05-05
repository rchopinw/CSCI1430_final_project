from DataProcessing import TFRecordData
from ProcessARGS import TFRecordConfigArgs

if __name__ == '__main__':
    print("Initializing TF Record Data Writer...")
    train_validation_tfr = TFRecordData(
        directory=TFRecordConfigArgs.train_directory,
        save_directory=TFRecordConfigArgs.train_save_directory,
        size=TFRecordConfigArgs.train_size,
        image_size=TFRecordConfigArgs.image_size
    )
    print("TF Record Data Writer Initialized Successfully...")
    print("Converting and Writing...")
    train_validation_tfr.convert()
    print("Done!")

    print("Configuring test data...")
    test_tfr = TFRecordData(
        directory=TFRecordConfigArgs.test_directory,
        save_directory=TFRecordConfigArgs.test_save_directory,
        size=TFRecordConfigArgs.test_size,
        image_size=TFRecordConfigArgs.image_size
    )
    test_tfr.convert()
    print("Done!")
