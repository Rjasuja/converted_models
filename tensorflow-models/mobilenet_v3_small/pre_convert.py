import argparse
import tensorflow as tf

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    tf.keras.backend.set_image_data_format('channels_last')

    model = tf.keras.applications.MobileNetV3Small(
        weights='./weights_mobilenet_v3_small_224_1.0_float.h5'
    )
    model.save(filepath='./mobilenet_v3_small_224_1.0_float.savedmodel')


if __name__ == '__main__':
    main()
