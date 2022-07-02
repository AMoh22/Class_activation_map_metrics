# metrics

metrics is a Python library that evaluate the quality of saliency maps.

1. Localization error (LE)
2. Dice coefficient (DC)
3. Energy pointing game (EPG)
4. Saliency map (SM)
5. Absolute mean error (AME)


## Usage

```python
import metrics

# returns the overlap error of predicted_box with the ground_truth_box
metrics.LE(ground_truth_box, predicted_box)

#...
```
