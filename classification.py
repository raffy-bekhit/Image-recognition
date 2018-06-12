from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy
import random

classes = []
seed=2

def read_data():
    "reads data from files and returns an array of the data"
    filename = 'segmentation.data'
    raw_data = open(filename, 'rt') #data as string
    reader = csv.reader(raw_data , delimiter=',')
    reader=list(reader) #list of data each value is seperated by ,
    fields = reader[0] # get the features names
    #fields.insert(0,'class')
    Xdata = reader[1:]
    raw_data.close()

    #reading second file
    filename = 'segmentation.data'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',')
    reader = list(reader)
    raw_data.close()
    Xdata = Xdata + reader[1:]

    labels=[] #labels of the data (Classes)

    for entry in Xdata: #loop the entries of the data make each value a float instead of string
        for i in range(0,len(fields)+1):
            if(i!=0):
                entry[i] = float(entry[i]) #convert to float
            else:

                if(entry[i] not in classes):
                    classes.append(entry[i]) #fill classes list which contains unique classes
                labels.append(classes.index(entry[i])) # lable of each entry is saved in this list
        entry.pop(0) #remove label from data


    return Xdata , labels , fields



def scale_list(list):
    scaled_data_list = list[:]
    scaled_data_list = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(list)
    return scaled_data_list

def split_data(array,labels,seed):
    "splits data into 50% for training and 50% for the test randomly"
    class_items = {} # {class: number of items in data}
    class_counter = {} #{class: counter for which item of this class is being accessed now}

    training_data = []
    test_data =[]

    test_labels=[] #labels of test data
    training_labels=[] #labels of training data


    i=0
    for item in classes:
        # initialize dictionaries
        class_items[item]=0
        class_counter[item]=0

    for label in labels:
        #count the the total items in the data of each label
        c = classes[label]
        class_items[c] = class_items[c] + 1

    j=0
    random_sequences = {} #dictionary for random sequences of each label
    for c in classes:
        #fill random_sequences

        random.seed(seed+j)
        random_sequences[c] = random.sample(range(class_items[c]),int(0.5*class_items[c]))
        random_sequences[c].sort()
        j+=5


    for i in range(len(array)):
        entry = array[i]
        c = classes[labels[i]]
        #splits data in a random way using random sequences
        if(len(random_sequences[c])>0 and class_counter[c]==random_sequences[c][0]):
            training_data.append(list(entry))
            training_labels.append(labels[i])
            random_sequences[c].pop(0)
        else:
            test_data.append(list(entry))
            test_labels.append(labels[i])
        class_counter[c]+=1
        i+=1

    return numpy.asarray(training_data) ,numpy.asarray( test_data) ,training_labels , test_labels


def accuracy(generated_labels,given_labels):
    "calculates the accuracy of a modeol by comparing the generated labels with the given labels"
    correct = 0
    for i in range(len(test_labels)):
        if (generated_labels[i] == test_labels[i]):
            correct= correct + 1
    return correct/len(given_labels)

def naive_bayes(training_data, training_labels):
    "returns accuracy"
    gnb = GaussianNB()
    gnb = gnb.fit(training_data, training_labels)
    generated_labels = gnb.predict(test_data)
    return accuracy(generated_labels,training_labels)

def decision_tree(training_data,training_labels):
    dt = DecisionTreeClassifier(criterion="entropy") #'entropy makes information gain the measure of the impurity'
    dt = dt.fit(training_data,training_labels)
    generated_labels = dt.predict(test_data)
    return accuracy(generated_labels,training_labels)

def knn(training_data,training_labels,k):
    model = KNeighborsClassifier(n_neighbors=k)
    model = model.fit(training_data,training_labels)
    generated_labels = model.predict(test_data)
    return accuracy(generated_labels,training_labels)

def print_stat(training_data,training_labels):
    n = naive_bayes(training_data,training_labels)*100
    d = decision_tree(training_data,training_labels)*100
    k3 = knn(training_data,training_labels,3)*100
    k10 = knn(training_data,training_labels,10)*100
    k20 = knn(training_data,training_labels,20)*100
    print("Naive Bayes Accuracy: %.2f%%" %n )
    print("Decision Tree Accuracy: %.2f%%" % d)
    print("KNN with k=3: %.2f%%" % k3)
    print("KNN with k=10: %.2f%%" % k10)
    print("KNN with k=20: %.2f%%" % k20)

data_list , labels ,fields= read_data() #list # fields is list of fields names

data_array = numpy.asarray(data_list)
scaled_data_array = scale_list(data_array)

print("With preprocessing:")
training_data ,  test_data ,training_labels ,test_labels= split_data(scaled_data_array,labels,seed)
print_stat(training_data,training_labels)

no_preproc_train_data , no_preproc_test_data ,no_preproc_train_labels, no_preproc_test_labels =  split_data(data_array,labels,seed)
print("")
print("")
print("Without preprocessing:")
print_stat(no_preproc_train_data,training_labels)
