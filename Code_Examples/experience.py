import metrics
import metrics_utils as utils
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms as transforms
import os
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.models as models
import numpy as np
import sys

val_json = sys.argv[1]
IMG_EXTENSIONS = sys.argv[2]

def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()

##Loader to get the ground truth informations
class ImagePathDataset(VisionDataset):
    

    def __init__(self, config, transform=None, target_transform=None,loader=default_loader, return_paths=False):
        
        super().__init__(root=config["root"], transform=transform,target_transform=target_transform)
        self.config = config

        self.loader = loader
        self.extensions = IMG_EXTENSIONS

        self.classes = config["classes"]
        self.class_to_idx = config["class_to_idx"]
        self.samples = config["samples"]
        self.targets = [s[1] for s in self.samples]
        self.return_paths = return_paths

    
    def __getitem__(self, index):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        name_m = os.path.basename(path).split('.')[0] + '.png'
        name_path_m = os.path.dirname(path).split('/')
        name_path_m[0] = '/'
        name_path_m[6] = 'validation-segmentation'
        path_m = os.path.join(os.path.join(*name_path_m),name_m)
        mask_gt = self.loader(path_m)
        if self.transform is not None:
            sample = self.transform(sample)
            mask_gt = self.transform(mask_gt)
        if self.target_transform is not None:
            target = self.target_transform(target)
        mask_gt = mask_gt[:,:,1]*256 + mask_gt[:,:,0]
        output = sample, target, mask_gt

        if self.return_paths:
            
            return output, path
        
        else:
             return output

    def __len__(self):
        
        return len(self.samples)

    @classmethod
    def from_path(cls, config_path, *args, **kwargs):
        
        return cls(config=json.loads(read_file(config_path)), *args, **kwargs)


transform = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         ])

val_loader = torch.utils.data.DataLoader(
             ImagePathDataset.from_path(
                 config_path = val_json,
                 transform=transform,
                 return_paths=True
                 ),
             batch_size=32, shuffle=False,
             num_workers=1, pin_memory=True
             )

model = models.resnet50(pretrained=True)
target_layer = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layer, use_cuda= False)
transform = transforms.Compose([
    transforms.ToTensor(),
])

used_metrics = ["LE", "OM","EPG", "DC","Pixel-wise-F1", "AME","SM"]


box_metrics_file = open('Box_metrics.csv', 'w')
mask_metrics_file = open('mask_metrics.csv', 'w')


#Data frame columns
box_metrics_file.write(",".join(used_metrics) + "\n")
mask_metrics_file.write(",".join(used_metrics) + "\n")

def get_predicted_mask(saliency_map, threshold):
    
    #The result of threholding the saliency map
    bin_map = utils.binarize(saliency_map, threshold)
    
    #Find the biggest component
    return utils.getLargestCC(bin_map)
    

#Metrics processing 
for i, ((images, labels, masks), paths) in enumerate(val_loader):

    targets = [ClassifierOutputTarget(labels[j]) for j in range(labels.shape[0])]

    saliency_map = cam(input_tensor=images, targets = targets)


    for k in range(len(images)):


        logits = model(images[k:k+1])

        predicted_label = torch.argmax(logits)

        saliency_average = np.mean(saliency_map[k])
    
        predicted_mask = get_predicted_mask(saliency_map[k], saliency_average)
    
        predicted_box = utils.get_bounding_box(predicted_mask)
    
        ground_truth_box = utils.get_bounding_box(masks[k])
    
        #Localization error
        
        #Box localization value
        ble = metrics.LE(ground_truth_box, predicted_box)

        box_metrics_file.write(str(ble)+",")

        #Mask localization error value
        mle = metrics.LE(masks[k], predicted_mask)
    
        mask_metrics_file.write(str(mle)+",")

        #Official metric
        if(labels[k] == predicted_label):

            box_metrics_file.write(str(ble)+",")
            mask_metrics_file.write(str(mle)+",")

        else:

            box_metrics_file.write(str(0)+",")
            mask_metrics_file.write(str(0)+",")

    
        #Energy pointing game
        box_metrics_file.write(str(metrics.EPG(ground_truth_box, saliency_map[k]))+",")
    
        mask_metrics_file.write(str(metrics.EPG(masks[k], saliency_map[k]))+",")

    
        #Dice coefficient equivalent to F1 score
        box_metrics_file.write(str(metrics.DC(ground_truth_box, predicted_box))+",")
    
        mask_metrics_file.write(str(metrics.DC(masks[k], predicted_mask))+",")
    
        #Pixel-wise F1-score
        box_metrics_file.write(str(metrics.pixel_wise_F_score(ground_truth_box, predicted_box))+",")
    
        mask_metrics_file.write(str(metrics.pixel_wise_F_score(masks[k], predicted_mask))+",")
    
        #Average mean error
        box_metrics_file.write(str(metrics.pixel_wise_F_score(ground_truth_box, saliency_map[k]))+",")
    
        mask_metrics_file.write(str(metrics.pixel_wise_F_score(masks[k], saliency_map[k]))+",")

        #Saliency metric

        soft = torch.nn.functional.softmax(logits, dim = 1)[0]
    
        ground_truth_probability = soft[labels[k]]

        box_metrics_file.write(str(metrics.SM(predicted_box, ground_truth_probability))+"\n")

        mask_metrics_file.write(str(metrics.SM(predicted_mask, ground_truth_probability))+"\n")



box_metrics_file.close()
mask_metrics_file.close()
