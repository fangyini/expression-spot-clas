import sys
import argparse
from load_images import *
from load_label import *
from extraction_preprocess import *
# Import the os module
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
print('Current working directory: {0}'.format(os.getcwd()))


def main(config):
    # Define the dataset and expression to spot
    dataset_name = config.dataset_name
    expression_type = config.expression_type
    train = config.train
    show_plot = config.show_plot

    print(' ------ Spotting', dataset_name, expression_type, '-------')

    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos, videoNames = load_images(dataset_name)

    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_videos, final_subjects, final_samples, final_names = load_gt(dataset_name, expression_type,
                                                                                     images,
                                                                                     subjectsVideos, subjects,
                                                                                     codeFinal,
                                                                                     videoNames)
    print('total subjects: ', len(subjects))
    print('total videos: ', len(images))
    x = 0
    for i in range(len(final_samples)):
        for j in range(len(final_samples[i])):
            x += len(final_samples[i][j])
    print('related emotions: ', x)
    print('related subjects ', len(final_subjects))
    print('related videos: ', len(final_images))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq')  # Specify CASME_sq or SAMMLV only
    parser.add_argument('--expression_type', type=str,
                        default='micro-expression')  # Specify micro-expression or macro-expression only
    parser.add_argument('--train', type=bool, default=True)  # Train or use pre-trained weight for prediction
    parser.add_argument('--show_plot', type=bool, default=True)

    config = parser.parse_args()
    main(config)
