import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    true_pred_args = np.where(prediction == True)[0]
    true_grnd_args = np.where(ground_truth == True)[0]
    true_intersection = np.in1d(true_pred_args, true_grnd_args)
    
    TP = true_intersection[true_intersection == True].shape[0]
    FP = prediction[prediction == True].shape[0] - TP
    
    accuracy = np.bincount(prediction == ground_truth)[1] / prediction.shape[0]
    precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0 if ground_truth[ground_truth == True].shape[0] == 0 else TP / ground_truth[ground_truth == True].shape[0]
    
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = np.bincount(prediction == ground_truth)[1] / prediction.shape[0]
    return accuracy
