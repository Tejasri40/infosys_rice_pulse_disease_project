**Model Training & Testing**

This milestone focuses on training and testing deep learning models for crop disease detection using image datasets.
Two separate models were trained for Rice and Pulse (Bean) crops using Convolutional Neural Networks (ResNet-18).

**Datasets Used**
1) Pulse Dataset
   
bean_rust, healthy, leaf_spot

 Number of classes: 3

2) Rice Dataset
   
blast, blight, tungro

 Number of classes: 3

**Training Process**
Dataset loaded using ImageFolder
Data split into training and validation sets
Model trained for multiple epochs
Validation accuracy monitored at each epoch
Best performing model saved

**Sample training output:**
Epoch 1/20
Train Loss: 10.46 | Train Acc: 71.05%
Val Loss: 10.16 | Val Acc: 52.08%
✓ Best model saved!


**Model Testing**
Trained models were tested using validation data
Class predictions and confidence scores verified
Model performance confirmed before deployment 

**Sample Output**
Loading trained model...
Model loaded successfully
Classes: ['blast', 'blight', 'tungro']
Testing multiple images...
Image: blast_001.jpg
Prediction: blast (99.75%)

**Outcome**
Successfully trained Rice and Pulse disease detection models
Verified prediction accuracy on validation data
Saved best-performing models for deployment
Prepared models for integration with Streamlit application



