# Advanced house price prediction 
## Goal
- predict house prices using machine learning
- Compare multiple regression models
- Find which makes the price to increase and where the model fail

## Key Questions
- Which models perform the best overall?
- Which features consistently matter?
- Do models behave differentlt by location or price range?


## Notebooks
There are four main notebooks in this project.
1. Data cleaning
2. Exploratory data analysis
3. Feature engineering
4. Modeling

## Workflows
1. First , we cleaned the data ( handling missing values,standardize the columns names, check duplicates, convert data types of zipcodes and date columns) and then saved the cleaned data set for the further steps.
2. For the eda, we only used the cleaned data set. There are 5 main parts in EDA ( Target variable, size, quality, location , correlation). After analyzing EDA, we have known that size, quality and location are the most important factors that influenced the house price.
3. The next part is the feature engineering, which is transforming the raw data into the meaningful data for the modelling. Therefore, we create the log price which will be the target variable, convert date into useful features, create house age, renovation features, meaningful features, reduce multicollinearity. Then define x and y, and encoding , then we got the feature engineering data set for the models.
4. Finally, we compare the four models ( Linear regression, Ridge regression, Random forest, XGB boost). Among these four models, XGB boost is the best model which get the lower RMSE and higher R squared.

## Library Used
streamlit
pandas
numpy
scikit-learn
xgboost
joblib


## Live Demo ( Streamlit app)
https://03adavancedhousepriceprediction-eeidjair6pz7raf8jhjumc.streamlit.app/

