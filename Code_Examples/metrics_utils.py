import numpy as np
from skimage.measure import label
import tensorflow as tf
import io
from PIL import Image

#Binarize a saliency map using a threshold
def binarize(saliency_map, threshold):
    
    #Get the dimensions of the saliency map
    h, w = saliency_map.shape
    
    #The binarized version of the saliency map
    result = np.zeros((h,w))
    
    for i in range(h):
        
        for j in range(w):
            
            if(saliency_map[i][j] >= threshold):
                
                result[i][j] = 1
                
    return result

#Donne la valeur moyenne des pixels dans une carte de chaleur
def get_average(saliency_map):
    
    #Les dimensions de la carte de chaleur
    n, m = saliency_map.shape
    
    #La moyenne au départ elle est à 0
    average = 0
    
    #Je somme pour chaque pixel
    for i in range(n):
        
        for j in range(m):
            
            average += saliency_map[i][j]
       
    # ma moyenne c'est la somme divisé par le nombre de pixels
    return average/(n*m)

#Permet de trouver la plus grande composante connexe dans une image en forme binnaire (les pixels ont valeur de 0 ou 1)
def getLargestCC(binarized):
    
    #Je labelise chaque composante dans l'image (je leurs associe une étiquette)
    labels = label(binarized)
    
    #Et je cherche la plus grande (celle qui a le plus de pixels)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=binarized.flat))
    
    return largestCC

#Retourne la plus grande boite qui entoure la composante connexe
def get_bounding_box(image):
    
    #Dimension de l'image
    n, m = image.shape
    
    #La coordonnée la plus haute
    upper = 0
    
    #La plus petite coordonnée dans l'image
    lower = 0
    
    #La coordonnée la plus à gauche
    left_result = n
    
    #La coordonnée la plus à droite
    right_result = 0
    
    #Pour savoir si la première bande supérieur a été trouvée
    first_bound = True
    
    #Pour chaque ligne
    for i in range(n):
        
        #Je cherhce le pixel le plus à gauche
        left = n
        
        #et le pixel le plus à droite
        right = 0
        
        #et je regarde si la colonne contient au moins un pixel
        contain_pixel = False
        
        #Pour chaque colonne
        for j in range(m):
            
            #Si l'image contient un pixel à 1
            if(image[i][j]) :
                
                #Je sais que je suis à la première colonne
                contain_pixel = True
                
                #Je compare quel pixel et le plus à gauche
                if(j < left):
                
                    left = j
                    
                if(j > right):
                    
                    right = j
                    
        if(left < left_result):
            
            left_result = left
            
        if(right > right_result):
            
            right_result = right
        
        if(contain_pixel and first_bound):
            
            upper = i
            first_bound = False
            
        elif(contain_pixel and not first_bound):
            
            lower = i
            
    return [upper, left_result], abs(left_result-right_result), abs(upper-lower)

def make_matrix(mask, coordinates):
    
    result = np.zeros(mask.shape)
    
    for i in range(coordinates[0][0], coordinates[0][0]+coordinates[2]):
        
        for j in range(coordinates[0][1], coordinates[0][1]+coordinates[1]):
            
            result[i][j] = 1
            
    return result

           
def read_trimap(path):
    
    with tf.io.gfile.GFile(path, 'rb') as fid:
        
        encoded_mask_png = fid.read()
    
    encoded_png_io = io.BytesIO(encoded_mask_png)
    
    return np.array(Image.open(encoded_png_io))

#Transform a trimap like the one above into a binary one 
def transform_trimap(trimap):
    
    h, w = trimap.shape
    
    result = []
    
    for i in range(h):
        
        line  = []
        
        for j in range(w):
            
            if(trimap[i][j] == 2):
                
                line.append(0)
                
            else:
                
                line.append(1)
                
        result.append(line)
                
    return np.array(result)
