from DataProcessing import TFRecordData
from ARGS import ARGS

if __name__ == '__main__':
    print("Initializing TF Record Data Writer...")
    tfr = TFRecordData(
        **ARGS.TFRecordConfig
    )
    print("TF Record Data Writer Initialized Successfully...")
    print("Converting and Writing...")
    tfr.convert()
    print("Done!")