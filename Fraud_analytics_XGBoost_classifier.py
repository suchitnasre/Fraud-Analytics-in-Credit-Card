# Fraud Analytics with XG Boost classifier

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv("creditcard.csv")

# Find total counts and perecentage of Fraud and Non Fraud transactions
Fraud_counts = len(data[data["Class"]==1])
Non_Fraud_counts = len(data[data["Class"]==0])
print("% of Fraud transactions:", Fraud_counts/(Fraud_counts+Non_Fraud_counts)*100)
print("% of Non Fraud transactions:", Non_Fraud_counts/(Fraud_counts+Non_Fraud_counts)*100)

# Plot the counterplot for Fraud and non Fraud transactions against Amount involved
Fraud_Transactions = data[data["Class"]==1]
Non_Fraud_Transactions = data[data["Class"]==0]
plt.subplot(121)
Fraud_Transactions[Fraud_Transactions["Amount"]<=2500].Amount.plot.hist(title = "Fraud Transactions")
plt.subplot(122)
Non_Fraud_Transactions[Non_Fraud_Transactions["Amount"]<=2500].Amount.plot.hist(title = "Non Fraud Transactions")

# Feature scaling
from sklearn.preprocessing import StandardScaler
data["Amount"] = StandardScaler().fit_transform(data.Amount.values.reshape(-1,1))


###################### Split the data into Training and Testing dataset ###################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.Class, test_size= 0.3, stratify = data.Class, random_state = 42)
# use stratify in train test split to proportion the unbalanced data carefully


############ Import the unbalanced data package and create the SMOTE function #######################
from imblearn.over_sampling import SMOTE
os = SMOTE(ratio= 0.3, random_state=42, k_neighbors= 5, m_neighbors=10, n_jobs=-1)
os_X_train, os_y_train = os.fit_sample(X_train, y_train)
print("Length of Training data: ", len(os_X_train))
print("Length of Fraud data: ", sum(os_y_train))
print("Length of Normal data: ", sum(os_y_train==0))
print("Prop. of Fraud data: ", sum(os_y_train)/len(os_X_train)*100)
print("Prop. of Normal data: ", sum(os_y_train==0)/len(os_X_train)*100)


## Fit the XGBoost model on training dataset and predict the data on testing dataset
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=200, random_state=42)
clf.fit(os_X_train, os_y_train)

pred = clf.predict(X_test.values)
prob = clf.predict_proba(X_test.values)

# Analyze the performance of model using Confusion matrix and classification report
# Achive the higher Precision and higher Recall
from sklearn.metrics import confusion_matrix, classification_report
cm =confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("----------------classification report---------------------------")
print(classification_report(y_test, pred))


## Compute and plot Average Precison-Recall
from sklearn.metrics import average_precision_score, precision_recall_curve
average_precision = average_precision_score(y_test, prob[:,1])
print("Average precision Recall score: ",(average_precision))
        
precision, recall, threshold = precision_recall_curve(y_test, prob[:,1])
        
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


## Compute and plot AUC-ROC curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
fpr, tpr,threshold_ = roc_curve(y_test, prob[:,1])
roc_auc = auc(fpr, tpr)                # the value is same as roc_auc_score in built function
        
plt.figure(figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr, tpr, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()
