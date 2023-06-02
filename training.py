import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from model.trainer import getDataloader
import torch
import os
from model.train_with_pytorch import train_with_pytorch
from model.transformer import Multitask_transformer
from sklearn.utils import class_weight

seed=666
random.seed(seed)
np.random.seed(seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pseudo_labeling(final_images, final_samples, k):
    pseudo_y = []
    video_count = 0 
    
    for subject in final_samples:
        for video in subject:
            samples_arr = []
            if (len(video)==0):
                pseudo_y.append([0 for i in range(len(final_images[video_count])-k)]) #Last k frames are ignored
            else:
                pseudo_y_each = [0]*(len(final_images[video_count])-k)
                for ME in video:
                    samples_arr.append(np.arange(ME[0]+1, ME[1]+1))
                for ground_truth_arr in samples_arr: 
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index+k) 
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr))/len(np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1 
                pseudo_y.append(pseudo_y_each)
            video_count+=1
    
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print('Total frames:', len(pseudo_y))
    return pseudo_y
    
def loso(dataset, pseudo_y, final_images, final_samples, k):
    #To split the dataset by subjects
    y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = y.copy()
    prevIndex = 0
    countVideos = 0
    
    #Get total frames of each video
    for video_index in range(len(final_images)):
      videos_len.append(final_images[video_index].shape[0]-k)
    
    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
      countVideos += len(final_samples[video_index])
      index = sum(videos_len[:countVideos])
      groupsLabel[prevIndex:index] = video_index
      print('Subject', video_index, ':', prevIndex, '->', index)
      prevIndex = index
    
    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(y))
    return X, y, groupsLabel


def spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p, show_plot, path):
    prev=0
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        print('Video:', countVideo+videoIndex)
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        score_plot_agg = score_plot.copy()
        
        #Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x+k] = score_plot[x:x+2*k].mean()
        score_plot_agg = score_plot_agg[k:-k]
        
        #Plot the result to see the peaks
        #Note for some video the ground truth samples is below frame index 0 due to the effect of aggregation, but no impact to the evaluation
        if(show_plot):
            plt.figure(figsize=(15,4))
            plt.plot(score_plot_agg) 
            plt.xlabel('Frame')
            plt.ylabel('Score')
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg[:,0], height=threshold[0], distance=k, 
                plateau_size=[0,2], width=1)
        if(len(peaks)==0): #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
            preds.append([0, 0, 0, 0, 0, 0]) 
        for peak in peaks:
            preds.append([peak-k, 0, peak+k, 0, 0, 0]) #Extend left and right side of peak by k frames
        for samples in video:
            gt.append([samples[0]-k, 0, samples[1]-k, 0, 0, 0, 0])
            total_gt += 1
            if(show_plot):
                plt.axvline(x=samples[0]-k, color='r')
                plt.axvline(x=samples[1]-k+1, color='r')
                plt.axhline(y=threshold, color='g')
        if(show_plot):
            #plt.show()
            plt.savefig(path+'/img_'+str(countVideo+videoIndex)+'.png')
            np.save(path+'/arr_'+str(countVideo+videoIndex)+'.npy', score_plot_agg)
        prev += len(dataset[countVideo+videoIndex])
        metric_fn.add(np.array(preds),np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
    return preds, gt, total_gt


def evaluation(preds, gt, total_gt, metric_fn): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN

def getSampleWeight(y_train, y_test, ratio):
    class_sample_count = np.unique(y_train, return_counts=True)[1]
    class_sample_count[0] = int(class_sample_count[0] * ratio)
    weight = 1. / class_sample_count
    samples_weight_train = weight[y_train]
    samples_weight_test = weight[y_test]
    return samples_weight_train, samples_weight_test

'''def delete_label_zero(y_train, ratio, windeoLen):  # list
    print('y train size: ', len(y_train))
    print('y train one label size: ', np.sum(y_train))
    length = len(y_train) - windeoLen
    zero_label = []
    one_label = []
    for idx in range(length):
        if np.sum(y_train[idx:(idx+windeoLen)]) == 0:
            zero_label.append(idx)
        else:
            one_label.append(idx)
    new_label = random.sample(zero_label, int(len(zero_label) * ratio))
    print('negative label len: ', len(new_label))
    print('positive label len: ', len(one_label))
    new_label.extend(one_label)
    return new_label'''

def training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset, train, show_plot,
            window_length, disable_transformer, step, threshold, batch_size, epochs):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groupsLabel)
    subject_count = 0
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    p = threshold #From our analysis, 0.55 achieved the highest F1-Score
    ratio = 0.5

    for train_index, test_index in logo.split(X, y, groupsLabel): # Leave One Subject Out
        subject_count+=1
        print('Subject : ' + str(subject_count))
        
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] #Get training set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index] #Get testing set
        
        print('------Initializing the model-------') #To reset the model at every LOSO testing
        
        path = 'Model_Weights/' + dataset_name + '/' + expression_type + '/s' + str(subject_count) + '/'
        if os.path.exists(path) == False:
            os.mkdir(path)
        samples_weight_train, samples_weight_test = getSampleWeight(y_train, y_test, ratio)
        test_dataloader = getDataloader(X_test, y_test, False, 1, window_length, samples_weight_test)
        model = Multitask_transformer(disable_transformer, num_encoder_layers=4, emb_size=400, nhead=4, dim_feedforward=512,
                                           dropout=0.1).float()
        model.to(DEVICE)
        if(train):
            # disable negative label removal
            #new_index = delete_label_zero(y_train, ratio, window_length)
            #random.shuffle(new_index)
            # todo: data agumentation
            train_loader = getDataloader(X_train, y_train, True, batch_size, window_length, samples_weight_train, step)
            train_with_pytorch(model, train_loader, test_dataloader, path, epochs)

        model.load_state_dict(torch.load(path + '/best'))
        model.eval()

        result = []
        for test_x, _, _ in test_dataloader:
            test_x = test_x.to(DEVICE)
            output = model(test_x)[0] # 2, 512
            result.extend(output)
        result = torch.stack(result).unsqueeze(1)
        result = result.cpu().detach().numpy()
        assert result.shape[0] == len(y_test)

        preds, gt, total_gt = spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p, show_plot,
                                       path)
        TP, FP, FN = evaluation(preds, gt, total_gt, metric_fn)
        
        print('Done Subject', subject_count)
        del X_train, X_test, y_train, y_test, result, model

    return TP, FP, FN, metric_fn

def final_evaluation(TP, FP, FN, metric_fn):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
    
    
# Result if Pre-trained weights are used, slightly different to the research paper

# Final Result for CASME_sq micro-expression
# TP: 18 FP: 327 FN: 39
# Precision =  0.0522
# Recall =  0.3158
# F1-Score =  0.0896
# COCO AP@[.5:.95]: 0.0069

# Final Result for CASME_sq macro-expression
# TP: 91 FP: 348 FN: 209
# Precision =  0.2073
# Recall =  0.3033
# F1-Score =  0.2463
# COCO AP@[.5:.95]: 0.0175

# Final Result for SAMMLV micro-expression
# TP: 41 FP: 323 FN: 118
# Precision =  0.1126
# Recall =  0.2579
# F1-Score =  0.1568
# COCO AP@[.5:.95]: 0.0092

# Final Result for SAMMLV macro-expression
# TP: 60 FP: 231 FN: 273
# Precision =  0.2062
# Recall =  0.1802
# F1-Score =  0.1923
# COCO AP@[.5:.95]: 0.0103
