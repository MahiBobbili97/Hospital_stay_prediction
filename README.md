<div id="top"></div>

<br />
<div align="center">
  
    

  <h2 align="center">Hospital Stay Prediction</h2>

  
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#pre_processing">Data PreProcessing</a></li>
    <li><a href="#smote">SMOTE</a></li>
    <li><a href="#Modeling">Modeling</a></li>
	<li><a href="#Results">Results</a></li>
  </ol>
</details>



## About The Project

Healthcare service is considered to be one of the most vital parts of modern human societies. The recent Covid-19 pandemic has stressed the importance of the role played by the healthcare services. In this project, we are going to work on a healthcare dataset that contains information about different patients, their conditions, and the number of days they were hospitalized. Our aim is to train different models that can predict the stay of a patient in a hospital, based on the initial conditions. This helps hospitals to identify patients of high length of stay risk at the time of admission and help in optimizing the treatment plan for patients who are likely to stay longer in the hospital and lower the chance of staff or visitor infection. We used Decision Tree classifier, Random Forest classifier, and CatBoost classifier to predict length of stay.


## Data PreProcessing

1)Imputing missing values.
Missing values can occur for a variety of causes, including observations that were not recorded or data tampering. Many machine learning methods do not accept data with missing values, therefore handling missing data is critical. The NA values are imputed using the mode of the column for any categorical variable in the dataset. After being transformed to a single hot vector, the categorical data column was combined with the original dataset comprising numeric columns. In this dataset we have missing null values in the ‘Bed Grade’ and ‘City_Code_Patient’ columns and replaced them with mode of the respective column.
2)Removing unwanted columns
‘case_id’ and ‘patientid’ were deleted from the data frame as they do not impact the classification.
3)Encoding categorical features to convert into numerical.
All input and output variables in machine learning models must be numeric. This means that if your data is categorical, you'll need to convert it to numbers before fitting and evaluating a model. LabelEncoder() and one-hot encoding are the two most used approaches. We used the LabelEncoder() method to encode into numerical labels.
4)Oversampling the data as the target feature ‘Stay’ is imbalanced.
When we examined the dataset, we discovered that the output variable is imbalanced. A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary. One way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model. An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be very effective.Perhaps the most widely used approach to synthesizing new examples is called the Synthetic Minority Oversampling Technique, or SMOTE. We'll utilize SMOTE from the imblearn package to balance the target variable.
5)Splitting the dataset.
The dataset is split using the train_test_split() method to split the data in such a way that the train data will be 80 percent and the test data will be 20 percent of the data. To split the dataset into train and test sets in a way that preserves the same proportions of examples in each class as observed in the original dataset we used the ‘stratify’ argument.


## SMOTE

SMOTE stands for Synthetic Minority Oversampling Technique
SMOTE is an oversampling technique that generates synthetic samples from the minority class.
It is used to obtain a synthetically class-balanced or nearly class-balanced training set, which is then used to train the classifier.


## Modeling

Decision Tree Classifier
Random Forest Classifier
CatBoost Classifier

## Results

Using both the oversampling approach and the k-fold, it was discovered that around 66 percent of the instances could be accurately categorized using the given data. When it comes to hospital administration, we can't expect precision. We initially had lower accuracy, as we were only able to categorize 38 percent of instances in predicting hospital stay length. While this is not a huge monetary value, it might make the difference between life and death for a patient by allowing them to reach a hospital with free beds available.

