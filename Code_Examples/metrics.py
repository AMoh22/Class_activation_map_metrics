#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:22:41 2022

@author: mohsine
"""

from math import log
from skimage.measure import label
import numpy as np

##The ground_truth (mask/box)  and the predicted (mask/box) must be in a binary format
##Please make sure that the ground_truth (mask/box) exists (the entry must be a matrix that contains at least a pixel with 1 value)
def IoU(ground_truth, predicted_mask):
    
    #verify if the two images has the same dimensions
    assert ground_truth.shape == predicted_mask.shape
    
    #Get the dimension of the picture
    m, n = ground_truth.shape
    
    #The variable that will contain the value of the intersection of the two masks area
    intersection = 0
    
    #The same but for union
    union = 0
    
    #We iterate on pixels
    for i in range(m):
        
        for j in range(n):
            
            #If the two of pixels is equal to one that mean that the pixel is in both masks
            if(int(ground_truth[i][j]) == 1 and int(predicted_mask[i][j]) == 1):
                
                intersection += 1
                union += 1
             
            #If the pixel is in one of the twos without being in the other we add 1
            if( int(ground_truth[i][j]) == 1 ^ int(predicted_mask[i][j]) == 1):
                
                union += 1
    
    return intersection/union

#Dice coefficient F1 score
#ground truth, preidcted --> (box/mask) must be numpy.array
def DC(ground_truth, predicted):

    #make sure that the dimensions are the same
    assert ground_truth.shape == predicted.shape

    #Get the dimensions of the matrix
    h, w = ground_truth.shape

    #The number of pixel with 1 as value that are in both of ground_truth and predicted mask or box
    TP = 0
    #The number of pixels that are in the ground_truth but not in the predicted mask or box
    TN = 0
    #The same but for predicted instead of ground truth
    FN = 0

    #We iterate in each pixel in both ground_truth and predicted to see what does match
    for i in range(h):

        for j in range(w):
            
            #We verify if they had 1 pixels in commun (intersection)
            if(int(ground_truth[i][j]) == 1 and int(predicted[i][j]) == 1):

                TP += 1

            #To count the number of pixels "1" that are in ground_truth but not in predicted
            elif(int(ground_truth[i][j]) == 1 and int(predicted[i][j]) == 0):

                FN += 1

            #Same thing but for predicted
            elif(int(ground_truth[i][j]) == 0 and int(predicted[i][j]) == 1):

                TN += 1

    #The dice formula (2*Union/(cardinality(ground_truth)+cardinality(predicted))
    return 2*TP/(2*TP+TN+FN)

#Localization error
def LE(ground_truth, prediction):
    
    return 1-IoU(ground_truth, prediction)


#Energy pointing game it is equivalent to the "Pixel-wise precision"
def EPG(ground_truth_mask, saliency_map):


    assert ground_truth_mask.shape ==  saliency_map.shape
    
    h, w = ground_truth_mask.shape
    
    #The saliency contained in the ground_truth (mask or box)
    ground_truth_saliency = 0
    
    #The saliency on the hole image
    global_saliency = 0
    
    
    for i in range(h):
        
        for j in range(w):
            
            #We and the value of the saliency when we have a pixel of 1 in the ground_truth
            ground_truth_saliency += float(saliency_map[i][j])*int(ground_truth_mask[i][j])
            
            #We sum the saliency of all pixels
            global_saliency += float(saliency_map[i][j])

    #The formula of energy pointing game
    return ground_truth_saliency/global_saliency


#The precision for the class of pixels with 1 value
def precision(ground_truth, predicted):
    
    #verify if the dimensions are the same
    assert ground_truth.shape == predicted.shape
    
    #Get the dimensions of one of the matrices
    h, w = ground_truth.shape
    
    #The well classified pixels
    good_predictions = 0
    
    #The number of pixels with value of one
    hypothesis = 0
    
    for i in range(h):
        
        for j in range(w):

            #If the predicted label correspond to the ground label
            if(ground_truth[i][j] == predicted[i][j] and ground_truth[i][j] == 1):
                
                good_predictions += 1
                
                hypothesis += 1
                
            #If it's not
            elif(predicted[i][j] == 1 and ground_truth[i][j] == 0):
                
                hypothesis += 1
                
    return good_predictions/hypothesis


#The recall of class 1 pixels
def recall(ground_truth, predicted):

    assert ground_truth.shape == predicted.shape
    
    h, w = ground_truth.shape
    
    #Well classified pixels
    good_predictions = 0
    
    #The number of pixels with 1 as value
    refrences = 0
    
    #Same thing as the precision but we calculate the number of references pixels of ground truth
    for i in range(h):
        
        for j in range(w):
            
            if(ground_truth[i][j] == predicted[i][j] and ground_truth[i][j] == 1):
                
                good_predictions += 1
                refrences += 1
                
            elif(ground_truth[i][j] == 1 and predicted[i][j] == 0):
                
                refrences += 1
                
    return good_predictions/refrences


#The beta F-score 
def F_score(ground_truth, predicted, beta=1):
    
    p = precision(ground_truth, predicted)
    
    r = recall(ground_truth, predicted)
    
    return (1+beta**2)*p*r/((beta**2)*p+r)


#Calculate the fraction of the ground truth  that is covered by the saliency map
def pixel_wise_recall(ground_truth, sallience_map):
    
    assert ground_truth.shape == sallience_map.shape
    
    m, n = ground_truth.shape
    
    ground_truth_saliency = 0
    
    refrences = 0
    
    for i in range(m):
        
        for j in range(n):
            
            if(int(ground_truth[i][j]) == 1):
                
                refrences += 1
                
                ground_truth_saliency += float(sallience_map[i][j])
                
    
    return ground_truth_saliency/refrences


#The F_beta score
def pixel_wise_F_score(ground_truth, saliency_map, beta=1):
    
    p = EPG(ground_truth, saliency_map)
    
    r = pixel_wise_recall(ground_truth, saliency_map)

    return (1+beta**2)*(p*r)/((beta**2)*p+r)


#Calculate the "1" class pixels in a binary mask or box
def get_cardinality(predicted):
    
    h, w = predicted.shape
    
    result = 0
    
    for i in range(h):
        
        for j in range(w):
            
            if(int(predicted[i][j]) == 1):
                
                result += 1
                
    return result

#The saliency metric
def SM(predicted, probability_class):
    
    h, w = predicted.shape
    
    return log( max( 0.05, get_cardinality(predicted) / (h*w) ) ) - log( probability_class )


#The absolute mean error
def AME(saliency_map, ground_truth):
    
    assert saliency_map.shape == ground_truth.shape
    
    h, w = saliency_map.shape
    
    result = 0
    
    for i in range(h):
        
        for j in range(w):
            
            result += abs(float(saliency_map[i][j]) - int(ground_truth[i][j]))
            
    return result/(h*w)

#The mask and box accuaracy it works for both of them
#Max value refer to sigma the second threshold
def accuaracy(ground_truths, saliency_maps, thresholds_list, max_value=0.5):
    
    assert len(ground_truths) == len(saliency_maps)
    
    results = np.zeros((len(thresholds_list),0))
    
    for i in range(len(ground_truths)):
        
        for j in len(thresholds_list):
            
            if(IoU(ground_truths[i], saliency_maps[i]) >= max_value):
                
                results[j] += 1
                
    results = results/len(ground_truths)  
    
    return np.max(results), results



def concept_influence(binary_segmentation_mask, top_k_binary_mask, k = 1_000):
    
    assert binary_segmentation_mask.shape ==  top_k_binary_mask.shape
    
    h, w = binary_segmentation_mask.shape
    
    top_k_binary_mask = 0
    
    relative_size_concept = 0
    
    for i in range(h):
        
        for j in range(w):
            
            top_k_binary_mask += binary_segmentation_mask[i][j]*top_k_binary_mask[i][j]
            
            relative_size_concept += binary_segmentation_mask[i][j]
    
    top_k_binary_mask /= k
    
    return top_k_binary_mask/relative_size_concept

def relevance_rank_accuracy(ground_truth, top_k_pixels):
    
    h, w = ground_truth.shape
    
    result = 0
    
    for pixel in top_k_pixels:
        
        result += ground_truth[pixel[0]][pixel[1]]
        
    return result/(h*w)
