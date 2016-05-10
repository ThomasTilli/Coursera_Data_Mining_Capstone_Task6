import math
import random
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,  GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import metrics

path="E:/online-kurse/coursera/Data Mining/capstone/task6/"
ftext="hygiene.dat"
fdatLables="hygiene.dat.labels"
flabel="label.txt"
fadd="hygiene.dat.additional"

def loadText():
   txt=[]
  
   with open(path+ftext,'r') as ft:
      for line in ft.readlines():
         txt.append(line)         
   print len(txt)
   return txt

def loadLabels():
   lab=[]
   i=0
   with open(path+fdatLables,'r') as fl:
      for line in fl.readlines():   
         if i==546: break
         lab.append(int(line))
         i+=1
   print len(lab)
   return lab

def loadAdditional():
   add=[]
  
   with open(path+fadd,'r') as fl:
      for line in fl.readlines():   
     
         add.append(line)
       
   print len(add)
   return add   


def combine(data,add):
   d=[]
  
   for i in range(0,len(data)):
      d.append(add[i]+' '+data[i])
     
   #print d[0:2]
   
   return d
   
   
def saveLabels(labels):
   with open(path+flabel,'w') as fl:
      fl.write('tomtom \n')
      for l in labels:
         fl.write(str(l)+'\n')
      
         
def categorize(data,lab):
   trainD=data[0:546]   
   testD=data[546:]   
   clf = Pipeline([('vect', CountVectorizer(stop_words='english',min_df=15,max_df=0.23,ngram_range=(1, 7))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                 ])   
   clf = clf.fit(trainD, lab)
   pred=clf.predict(trainD)   
   score = metrics.f1_score(lab,pred)
   print("f1-score:   %0.3f" % score)
   
   pred=clf.predict(testD)
   saveLabels(pred)   
 

def categorizeSGD(data,lab):
   trainD=data[0:546]   
   testD=data[546:]   
   clf = Pipeline([('vect', CountVectorizer(stop_words='english',min_df=15,max_df=0.22,ngram_range=(1, 7))),
                   ('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='log', penalty='elasticnet',l1_ratio=0.151, alpha=1.01e-3, n_iter=11,      random_state=42))       
                   ,   
   ])    
   clf = clf.fit(trainD, lab)
   pred=clf.predict(trainD)   
   score = metrics.f1_score(lab,pred)
   print("f1-score:   %0.3f" % score)
   
   pred=clf.predict(testD)
   saveLabels(pred)      
       
d=loadText()   
l=loadLabels()
a=loadAdditional()
ad=combine(d,a)

categorize(ad,l) 