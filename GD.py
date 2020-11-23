# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

from pyspark.mllib.regression import LabeledPoint
#from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    #feats.insert(0,label)#######################################################
    features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, features)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("/opt/bitnami/spark/data_banknote_authentication.txt")
parsedData = data.map(mapper)

#points = spark.read.text("/opt/bitnami/spark/data_banknote_authentication.txt") 

# Train model
iterations = 100
D = 4
#Initialization
w = 2*np.random.ranf(size=D) -1
print("Initial w: " + str(w))
#parsedData.foreach(print)

def gradient(matrix, w):
    #print("w = " + str(w))
    Y = matrix.label
    X = matrix.features
    return((1.0/(1.0+np.exp(-X.dot(w)))-Y) * X.T)
    

def add(x,y):
    return x+y

for i in range(iterations):
    #print("On iteration %i" % (i+1))
    #w -= parsedData.map(lambda m: gradient(m, w)).reduce(add)
    grad = parsedData.map(lambda m: gradient(m,w)).reduce(add)/parsedData.count()
    w = w-grad
print("Final w: " + str(w))

def predict(fea,w):
    Y = 1.0/(1.0+np.exp(-fea.dot(w)))
    return round(Y)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label), 
        predict(point.features,w)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda p: p[0] != p[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

