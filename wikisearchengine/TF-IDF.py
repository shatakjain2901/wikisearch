from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Loading documents (one per line).
rawData = sc.textFile("C:/Users/shatak/Desktop/shatak3rd/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])

#hashing the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# we have an RDD of sparse vectors representing each document,


# computing the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

# the article for "Abraham Lincoln" is in our data
# set, so let's search for "Gettysburg" (Lincoln famous speech there):

#figuring out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

#  extracting the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# zipping in the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames)

#  printing the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())
