# Semantic Segmentation

## Running the Code
run `python starter.py` to run the code with the basic fcn model using cross entropy loss and without data augmentations

To run the code using dice loss and data augmentation:
* Comment the line `self.transform_alb = None` in `dataloader.py`
* Uncomment the line `criterion = dice_loss` in `starter.py` 

To run the code using the other models:
* Comment `fcn_model = FCN(n_class=n_class)` in `starter.py`
* Uncomment the line corresponding to model you wish to use below the above line in `starter.py`
* Do the same thing in test while performing the function `test`
---

## Saving the plots
All plots will be saved in the folder plots/
