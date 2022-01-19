 STEPS TO REPRODUCE SUBMISSION
1. Download data - Plato_Radiant_Data_Download

2. Preprocess data to numpy arrays - Plato_Radiant_Data_Preprocessing

3. Train LGBM model - Plato_Lgbm
•	- For faster training, upload the Plato_Lgbm notebook to colab
•	- Upload the radiant pixels dataset and the sample submission file
•	- Enable TPU which has 40 cores
•	- Run all to get the LGBM_SUB file

4. Train Neural Network - Plato_Neural_Net
•	- Upload Plato_Neural_Net notebook to colab
•	-Upload the Plato_Neural_Net notebook_requirements.txt file
•	- Upload the radiant pixels dataset and sample submission file
•	- Enable GPU runtime
•	- Run all to get the pytorch_tabular file

5. Blend predictions from the two models - Plato_Blend_Predictions
•	- Upload the Plato_Blend_Predictions notebook to colab
•	- Upload the LGBM_SUB file and the pytorch_tabular file
•	- Run all to get the final submission file
Link to data used: https://drive.google.com/file/d/1vVP0ekUBPXrqG6vaoYb5R6MyyPIJL0pL/view?usp=sharing  


Note that the score will vary due to the nature of deep learning randomness irrespective of setting seed.

