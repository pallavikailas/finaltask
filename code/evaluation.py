import numpy as np
import pandas as pd
import warnings
import math

#Classification task 
#F1 score
# True labels and predicted labels
true_labels = [1, 0, 1, 1, 0, 1, 0, 0]
predicted_labels = [1, 1, 1, 0, 0, 1, 0, 1]
def calculate_f1_score(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    # Check if precision and recall are both 0 to avoid division by zero
    if precision == 0 or recall == 0:
        return 0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

# Example usage
true_positive = 0 
false_positive = 0
false_negative = 0
for i in range(len(true_labels)):
    if true_labels[i]==1 and predicted_labels[i] == 0:
        false_negative+=1
    elif true_labels[i]==0 and predicted_labels[i] == 1:
        false_positive+=1
    else:
        true_positive+=1


f1_score = calculate_f1_score(true_positive, false_positive, false_negative)
print("F1 Score:", f1_score)


#Regression task
#RMSE score
# True values and predicted values
true_values = [2.5, 3.0, 1.5, 4.0, 5.1]
predicted_values = [2.7, 2.9, 1.8, 3.9, 5.0]

# Calculate the squared differences
for i in range(len(true_values)):
    squared_diffs = (true_values[i] - predicted_values[i]) ** 2 

# Calculate the mean squared error (MSE)
mse = np.mean(squared_diffs)

# Calculate the RMSE
rmse = np.sqrt(mse)

print("RMSE:", rmse)
