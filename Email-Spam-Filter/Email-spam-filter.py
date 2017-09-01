import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    #  non-word removal here
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(5000)
    return dictionary

def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),5000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

train_dir = 'ling-spam\\train-mails'
dictionary = make_Dictionary(train_dir)
print (dictionary)
train_labels = np.zeros(702)
train_labels[351:701] = 1 #spam emails
train_matrix = extract_features(train_dir)
print(train_matrix[1])
# Training SVM and Naive bayes classifier
NB_model = MultinomialNB()
#GNB_model = GaussianNB()
#BNB_model = BernoulliNB()
LinearSVM_model = LinearSVC()
SVM_model = SVC()
NB_model.fit(train_matrix,train_labels)
#GNB_model.fit(train_matrix,train_labels)
#BNB_model.fit(train_matrix,train_labels)
LinearSVM_model.fit(train_matrix,train_labels)
SVM_model.fit(train_matrix,train_labels)

#use GridSearch to better choose tuning parameters C and gamma
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(train_matrix,train_labels)

print(grid.best_params_)
print(grid.best_estimator_)

# Test the unseen mails for Spam
test_dir = 'ling-spam\\test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1 #spam emails

NB_predictions = NB_model.predict(test_matrix)
#GNB_predictions = GNB_model.predict(test_matrix)
#BNB_predictions = BNB_model.predict(test_matrix)
LinearSVM_predictions = LinearSVM_model.predict(test_matrix)
grid_predictions = grid.predict(test_matrix)
SVM_predictions = SVM_model.predict(test_matrix)

print(confusion_matrix(test_labels,NB_predictions))
print(confusion_matrix(test_labels,LinearSVM_predictions))
print(confusion_matrix(test_labels,SVM_predictions))
print(confusion_matrix(test_labels,grid_predictions))
#print(confusion_matrix(test_labels,GNB_predictions))
#print(confusion_matrix(test_labels,BNB_predictions))
