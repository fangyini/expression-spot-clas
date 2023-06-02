import sys
import argparse
from data_process.load_images import *
from data_process.load_label import *
from data_process.extraction_preprocess import *
from training import *
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
    batch_size = config.batch_size
    epochs = config.epochs
    window_length = config.window_length
    disable_transformer = config.disable_transformer
    step = config.step
    
    print(' ------ Spotting', dataset_name, expression_type, '-------')

    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos, videoNames = load_images(dataset_name)
    
    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_videos, final_subjects, final_samples, final_names = load_gt(dataset_name, expression_type, images,
                                                                        subjectsVideos, subjects, codeFinal,
                                                                        videoNames)

    print('\n ------ Computing k ------')
    #k = cal_k(dataset_name, expression_type, final_samples)
    k = 6
    print('hardcoded k = 6')
    
    # Feature Extraction & Pre-processing
    print('\n ------ Feature Extraction & Pre-processing ------')
    dataset = extract_preprocess(final_images, k, final_names, dataset_name, fromSaved=False)
    
    # Pseudo-labeling
    print('\n ------ Pseudo-Labeling ------')
    pseudo_y = pseudo_labeling(final_images, final_samples, k)
    
    # LOSO
    print('\n ------ Leave one Subject Out ------')
    X, y, groupsLabel = loso(dataset, pseudo_y, final_images, final_samples, k)
    
    # Model Training & Evaluation
    # X: len, feature; y: len; groupLabel: len;
    print('\n ------ SOFTNet Training & Testing ------')
    train = True
    TP, FP, FN, metric_fn = training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset, train,
                                     show_plot, window_length, disable_transformer, step, threshold=0.7, batch_size=batch_size, epochs=epochs)
    final_evaluation(TP, FP, FN, metric_fn)
    print('previous p is: 0.7.')
    fortestingp = np.arange(0.75, 1, 0.05)
    for p in fortestingp:
        TP, FP, FN, metric_fn = training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset,
                                         False,
                                         show_plot, window_length, disable_transformer, step, threshold=p, batch_size=batch_size, epochs=epochs)
        final_evaluation(TP, FP, FN, metric_fn)
        print('previous p is: ' + str(p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq') # Specify CASME_sq or SAMMLV only
    parser.add_argument('--expression_type', type=str, default='micro-expression') # Specify micro-expression or macro-expression only
    parser.add_argument('--train', type=bool, default=True) #Train or use pre-trained weight for prediction
    parser.add_argument('--show_plot', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--window_length', type=int, default=12)
    parser.add_argument('--disable_transformer', type=bool, default=False)
    parser.add_argument('--step', type=int, default=5)
    # todo: add step, delete unused code, add token, add au,

    config = parser.parse_args()
    main(config)
