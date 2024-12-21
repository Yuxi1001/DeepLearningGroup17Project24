# DeepLearningGroup17Project24
Structured_SSM_for_EHR_Classification

Project 24 for 02456 2024

# Background
This repository allows you to train and test a variety of electronic health record (EHR) classification models on mortality prediction for the Physionet 2012 Challenge (`P12`) dataset. More information on the dataset can be found here (https://physionet.org/content/challenge-2012/1.0.0/). Note that the data in the repository has already been preprocessed (outliers removed, normalized) in accordance with https://github.com/ExpectationMax/medical_ts_datasets/tree/master and saved as 5 randomized splits of train/validation/test data. Adam is used for optimization.

# Create Environment
The dependencies are listed for python 3.9.

To create a venv, run: 

`pip install -r requirements.txt` 



# Run models 
All five models have been implemented in PyTorch and can be trained/tested on the P12 dataset. Due to the computational intensity, all calculations are performed on an HPC system. When running, please put the P12data folder inside the Structured_SSM folder.

For each model (EHRMamba and four baseline models), you only need to upload the corresponding cli_project_job_<modelName>.sh file to the HPC platform. Each script contains a unique set of hyperparameters, which can be adjusted directly within the script as needed. No additional modifications are required.


