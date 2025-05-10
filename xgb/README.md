# Symptoms Prediction Using XGBoost
A disease prediction tool using xgboost to implement.


# Dependency
### To use the tool
1. pandas
2. numpy
3. pickle
4. xgboost

### To train the model
5. sklearn
6. shutil
7. joblib


# How to Use?
1. Clone the repo and go to xgboost dir
2. Fill out the sample.csv. Fill in 1 if you have the symptom, otherwise 0.
3. ```python xgb_predict.py```
4. Check the predictions (xgb_predictions.csv)

# Output
The output looks like the table below:

| top5_prediction                                                                                                                                            | all_symptoms                                                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| panic disorder (0.997),<br/>drug abuse (methamphetamine) (0.001),<br/>anxiety (0.001),<br/>panic attack (0.000),<br/>acute respiratory distress syndrome (ards) (0.000) | anxiety and nervousness, shortness of breath,<br/>depressive or psychotic symptoms, chest tightness, palpitations,<br/>irregular heartbeat, breathing fast |
 ... | ... | 
