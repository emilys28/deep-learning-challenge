# deep-learning-challenge
Module 21

# Author 
Emily L Sims

# Report on the Neural Network Model
## Overview of the analysis: 
The purpose of my deep learning model was to create a binary classification model to help predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. My goal was to predict a binary outcome based on the input features provided in the dataset. The analysis involves preprocessing the data, defining a neural network model, compiling, training, and evaluating the model to achieve the desired performance.

## Results:
- Data Preprocessing
  - What variable(s) are the target(s) for your model? The target variable for the model is typically indicated by the column named IS_SUCCESSFUL, which represented whether an organization was successful (1) or not (0).
  - What variable(s) are the features for your model? The features for the model are all the other columns in the dataset, including APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.
  - What variable(s) should be removed from the input data because they are neither targets nor features? Variables such as EIN and NAME are identifiers and should be removed from the input data as they are neither targets nor features.
    
- Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why? The model consists of two hidden layers with 50 and 30 neurons, respectively. The first hidden layer uses the sigmoid activation function, and the second hidden layer uses the ReLU activation function. These choices were made based on experimentation and common practices in neural network architectures.
  - Were you able to achieve the target model performance? Yes, I was able to achieve an Accuracy of 1.0 and Loss of 2.250281072591065e-09. The accuracy of 1 tells us that the model correctly identified all samples in the test data. The very low loss value shows us that the predictions are extremely close to the real values. 
  - What steps did you take in your attempts to increase model performance? Since my model was already performing well, I just experimented by adding another hidden layer with an additional activition model, softmax. The 'optomized' model didn't perform as well as the first one. The loss was 0.010103730484843254 with an accuracy of 0.9986006021499634. 

## Summary:
These results suggest that the model has learned the patterns in the data very well and is making accurate predictions of the success of organizations based on various input features. However, achieving perfect accuracy on the test data may also indicate potential overfitting, especially if the model has not been evaluated on unseen data or if the dataset is relatively small. Further optimization and fine-tuning may be necessary to achieve the desired performance metrics. 

To ensure the robustness of the model, it's essential to validate its performance on unseen data, perhaps by employing techniques such as cross-validation or by splitting the data into separate training, validation, and test sets.

Additionally, considering the nature of the classification problem and the complexity of the dataset, alternative models such as ensemble methods (e.g., random forest, gradient boosting) or deep learning architectures like convolutional neural networks (CNNs) or recurrent neural networks (RNNs) could also be explored to potentially improve model performance. These models may better capture the intricate relationships within the data and provide more accurate predictions.

Overall, achieving high accuracy and low loss on the test data is a positive outcome, but it's important to interpret these results cautiously and consider potential sources of bias or overfitting.

# Requirements
## Preprocess the Data (30 points)
- Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset (2 points)
- Drop the EIN and NAME columns (2 points)
- Determine the number of unique values in each column (2 points)
- For columns with more than 10 unique values, determine the number of data points for each unique value (4 points)
- Create a new value called Other that contains rare categorical variables (5 points)
- Create a feature array, X, and a target array, y by using the preprocessed data (5 points)
- Split the preprocessed data into training and testing datasets (5 points)
- Scale the data by using a StandardScaler that has been fitted to the training data (5 points)

## Compile, Train and Evaluate the Model (20 points)
- Create a neural network model with a defined number of input features and nodes for each layer (4 points)
- Create hidden layers and an output layer with appropriate activation functions (4 points)
- Check the structure of the model (2 points)
- Compile and train the model (4 points)
- Evaluate the model using the test data to determine the loss and accuracy (4 points)
- Export your results to an HDF5 file named AlphabetSoupCharity.h5 (2 points)

## Optimize the Model (20 points)
- Repeat the preprocessing steps in a new Jupyter notebook (4 points)
- Create a new neural network model, implementing at least 3 model optimization methods (15 points)
- Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5 (1 point)

## Write a Report on the Neural Network Model (30 points)
- Write an analysis that includes a title and multiple sections, labeled with headers and subheaders (4 points)
- Format images in the report so that they display correction (2)
- Explain the purpose of the analysis (4)
- Answer all 6 questions in the results section (10)
- Summarize the overall results of your model (4)
- Describe how you could use a different model to solve the same problem, and explain why you would use that model (6)

# References
IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/
