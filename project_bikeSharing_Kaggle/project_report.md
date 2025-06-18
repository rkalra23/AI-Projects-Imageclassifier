# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Ratan Kalra

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
No negative value should be there. It should be set to zero if there is any negative value otherwise Kaggle will not allow you to submit it

### What was the top ranked model that performed?
WeightedEnsemble_L3

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
There were certain  columns which were  of int type but they were categorical. Also , converted the datetime column to day , month and weekday.

### How much better did your model preform after adding additional features and why do you think that is?
The score got significantly improved

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
In my case it was overfitting. when I tried to tune hyper parameter.

### If you were given more time with this dataset, where do you think you would spend more time?
EDA

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
	model	        hpo1	  hpo2	    hpo3	       score
	initial	       default	 default   default	        1.77
	add_features   default	 default   default	        0.49
	hpo	          num_trials scheduler  searcher	    0.61


### Create a line plot showing the top model score for the three (or more) training runs during the project.
completed

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

completed


## Summary
AutoGluon was used to predict bike sharing demand. Initially, the model required adjustments to ensure no negative predictions were submitted to Kaggle. The top-performing model was WeightedEnsemble_L3. Through exploratory data analysis, several integer features were correctly identified as categorical, and new time-based features (day, month, weekday) were created from the datetime column, which significantly improved the model's score from 1.77 to 0.49. However, hyperparameter tuning led to slight overfitting, resulting in a score of 0.61. Given more time, further improvements would focus on deeper EDA to uncover additional useful patterns.


