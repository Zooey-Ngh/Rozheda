
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
X = data.data
y = data.target
labels = data.target_names
feature_names = data.feature_names
# print(data.DESCR)
print(data.target_names)

#Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Add Gaussian noise to the data set
np.random.seed(42)
noise_factor = 0.5
X_noise = X_scaled + noise_factor *np.random.normal(loc=0.0,scale=1.0,size = X.shape)

#load both data
df = pd.DataFrame(X_scaled,columns=feature_names)
df_noisy = pd.DataFrame(X_noise, columns=feature_names)

#Display the first few rows
# print('Original Data (first 5 rows)')
# print(df.head())

# print('\nNoisy Data(first 5 rows)')
# print(df_noisy.head())
plt.figure(figsize = (12,6))
plt.plot(df[feature_names[5]], label='Original', lw = 3)
plt.plot(df_noisy[feature_names[5]], '--',label ='Noisy')
plt.title('Scaled feature comparison with and withou noise')
plt.xlabel(feature_names[5])
plt.legend()
plt.tight_layout()
# plt.show()

#Scatterplot
plt.figure(figsize=(12,6))
plt.scatter(df[feature_names[5]],df_noisy[feature_names[5]],lw =5)
plt.title('Scaled feature comparison with and whithout noise')
plt.xlabel('Original Data')
plt.ylabel("noisy data")
plt.tight_layout()
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X_noise,y, test_size=0.3, random_state=42)

#initialize the models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1,random_state=42)

#Fit the models to the trsining data
knn.fit(X_train, y_train)
svm.fit(X_train,y_train)

# Evaluate the models
#predict on the test set
y_pred_knn =knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Print the accuracy scores ans classification reports for both models
print(f'KNN test accuracy :{accuracy_score(y_test,y_pred_knn):.3f}')
print(f'SVM Testing Accuracy: {accuracy_score(y_test,y_pred_svm):03f}')

print('\nKNN testing data classification report:')
print(classification_report(y_test,y_pred_knn))

print('\nSVM Testing Data Classifiaction Report:')
print(classification_report(y_test, y_pred_svm))

#confusion matrix

conf_matrix_knn = confusion_matrix(y_test,y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test,y_pred_svm)

fig,axes = plt.subplots(1,2,figsize = (12,5))
sns.heatmap(conf_matrix_knn,annot=True, cmap='Blues',fmt='d',ax =axes[0], xticklabels=labels,yticklabels= labels)

axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True,cmap='Blues',fmt = 'd',ax =axes[1],xticklabels=labels,yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()