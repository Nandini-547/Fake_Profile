
import pickle
import numpy asnp
import pandas aspd
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
results = []
names =[]

def read_datasets():
real_users = pd.read_csv("users.csv")
fake_users = pd.read_csv("fusers.csv")
plt.xlabel('users')
plt.ylabel('total number of users')
plt.title('profile user analysis')
x = pd.concat([real_users,fake_users])
y = len(fake_users)*[0] + len(real_users)*[1]
print("number of real_users in the dataset is :"+str(len(real_users)))
print("number of fake_users in the datset is :"+str(len(fake_users)))
histogram = [len(real_users),len(fake_users)]
plt.hist(histogram,rwidth=0.95,color='r',orientation='horizontal')
plt.show()
return x,y
def predict_sex(name):
d = gender.Detector(case_sensitive=False)
first_name= str(name).split(' ')[0]
sex = d.get_gender(u"{}".format(first_name))
sex_code_dict = {'female': -2, 'mostly_female': -1,'unknown':0, 'andy': 0, 'mostly_male':1,
'male': 2}
code = sex_code_dict[sex]
return code
def extract_features(x):
lang_list = list(enumerate(np.unique(x['lang'])))
lang_dict = { name : i for i, name in lang_list }
x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)
x.loc[:,'sex_code'] = predict_sex(x['name'])

feature_columns_to_use =
['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_code','lan
g_code']
x = x.loc[:,feature_columns_to_use]
return x
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
target_names=['Fake','Genuine']
plt.imshow(cm, interpolation='nearest', cmap=cmap)
title = title
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
x,y = read_datasets()
x = extract_features(x)
print(x.head())
print(x.tail())
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
#support vector machine
svm_clf = svm.SVC(decision_function_shape='ovo')
svm_clf.fit(X_train, y_train)

Y_pred_svm = svm_clf.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm,y_test)*100,2)
print("The accuracy score achieved using svm is: "+str(score_svm)+" %")
cm=confusion_matrix(y_test,Y_pred_svm)
print("confusion matrix"+str(cm))
plot_confusion_matrix(cm,'svm')
results.append(score_svm)
names.append('svm')
#Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=50)
rf_classifier.fit(X_train, y_train)
train_predictions = rf_classifier.predict(X_train)
prediction = rf_classifier.predict(X_test)
err_training = mean_absolute_error(train_predictions, y_train)
err_test = mean_absolute_error(prediction, y_test)
print("The accuracy score achieved using Random forest is : {}".format(100 -
(100*err_training)))
cm=confusion_matrix(y_test, prediction)
print("confusion matrix"+str(cm))
plot_confusion_matrix(cm,'random_forest')
score_rf=100 - (100*err_training)
results.append(score_rf)
names.append('rf')
#Decision tree
classifierDT=DecisionTreeClassifier(criterion="gini", random_state=50, max_depth=20,
min_samples_leaf=10)
classifierDT.fit(X_train,y_train)

Y_pred_DT = classifierDT.predict(X_test)
score_DT = round(accuracy_score(Y_pred_DT,y_test)*100,2)
print("The accuracy score achieved using decision is: "+str(score_DT)+" %")
cm=confusion_matrix(y_test, Y_pred_DT)
print("confusion matrix"+str(cm))
plot_confusion_matrix(cm,'descision_tree')
results.append(score_DT)
names.append('DT')
#knn
knn_classifier = KNeighborsClassifier(n_neighbors = 16)
knn_classifier.fit(X_train,y_train)
Y_pred_KNN = knn_classifier.predict(X_test)
score_KNN = round(accuracy_score(Y_pred_KNN,y_test)*100,2)
print("The accuracy score achieved using Knn is: "+str(score_KNN)+" %")
cm=confusion_matrix(y_test,Y_pred_KNN)
print("confusion matrix"+str(cm))
plot_confusion_matrix(cm,'knn')
results.append(score_KNN)
names.append('KNN')
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
pickle.dump(rf_classifier, file)
batch_size=[16,32,64,128]
print(results)
print(names)
x_pos = np.arange(len(names))
plt.plot(x_pos,batch_size,results,'b-o');
plt.title(label='Comparision of algorithms')

plt.xticks(x_pos, names)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()