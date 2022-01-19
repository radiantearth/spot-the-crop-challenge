# Radiant Earth Spot the Crop Challenge

The solution is an ensemble of 2 main models, XceptionTime (https://arxiv.org/abs/1911.03803) and InceptionTime(https://arxiv.org/abs/1909.04939),
implemented in the tsai library. The models are trained on various datasets derived from the Sentinel-2 bands data. 

Two methods are to standardize the number of observations for each field:
 - Resample the original data to intervals of 3,5,7 and 10 days. Null values are imputed with zeros.
 - Group the 76 unique dates in the original data into 38 sets of adjacent dates; aggregate and average the data by field and date-set.

This results in 5 different versions of the original data.

4 XceptionTime models are trained on the 4 resampled datasets and 2 InceptionTime models are trained on the 5th dataset. 

Each model is trained with different combinations of bands and vegetation indices to increase variation and improve generalization of the final ensemble.

Refer to FEATURES in the documentation for the complete list of indices used.

With 5 fold stratified sampling, the final ensemble has 6*5 = 30 models.


Refer to EXPERIMENTS section in the documentation for training results.

### To reproduce 
Install requirements:

`pip install -r requirements.txt`

cd to the src directory run:

`chmod +x* .sh`

Then follow the steps and commands outlined in the TASKS sheet in the documentation to reproduce the results. 



### Environment
Google Colab Pro or similar with:
- Tesla P100 GPU
- Cuda Version 11
- 16GB RAM

### Notes

Data preprocessing tasks (0,1,2 for XceptionTime; 0,7 for InceptionTime),  must be run sequentially. 
Once preprocessing is done, training tasks (3-6 for XceptionTime; 8,9 for InceptionTime) can be run in any order.

To run inference only without training, unzip models.zip and run task 10.

