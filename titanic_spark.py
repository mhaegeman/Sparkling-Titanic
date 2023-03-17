# import libraries 
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import mean, count, col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

###########################################################################################################
###################################### CRETING SESSION & DATA IMPORT ######################################
###########################################################################################################

# create spark session
spark = SparkSession.builder.appName("Titanic Data Analysis").getOrCreate()

# dataset definition
titanic_schema = StructType([
    StructField("PassengerId", IntegerType(), nullable=False),
    StructField("Survived", IntegerType(), nullable=False),
    StructField("Pclass", IntegerType(), nullable=False),
    StructField("Name", StringType(), nullable=False),
    StructField("Sex", StringType(), nullable=False),
    StructField("Age", DoubleType(), nullable=True),
    StructField("SibSp", IntegerType(), nullable=False),
    StructField("Parch", IntegerType(), nullable=False),
    StructField("Ticket", StringType(), nullable=False),
    StructField("Fare", DoubleType(), nullable=False),
    StructField("Cabin", StringType(), nullable=True),
    StructField("Embarked", StringType(), nullable=True)
])

# Replace 'path/to/titanic.csv' with the actual path to your Titanic dataset
titanic_df = spark.read.csv('data/train.csv', header=True, schema=titanic_schema)

# showing the spark df
titanic_df.show()

###########################################################################################################
######################################## PREPROCESSING OF THE DATA ########################################
###########################################################################################################

# Calculate the mean age
mean_age = titanic_df.agg(mean(col("Age"))).collect()[0][0]

# Fill missing ages with the mean age
titanic_df = titanic_df.na.fill({"Age": mean_age})

# Calculate the most common 'Embarked' value
mode_embarked = titanic_df.groupBy("Embarked") \
    .agg(count("*").alias("count")) \
    .orderBy(col("count").desc()) \
    .collect()[0]["Embarked"]

# Fill missing 'Embarked' values with the most common value
titanic_df = titanic_df.na.fill({"Embarked": mode_embarked})

# Drop the 'Cabin' column
titanic_df = titanic_df.drop("Cabin")

# Convert 'Sex' to numerical values using StringIndexer
sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
titanic_df = sex_indexer.fit(titanic_df).transform(titanic_df)

# One-hot encode the 'SexIndex' column
sex_encoder = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")
titanic_df = sex_encoder.fit(titanic_df).transform(titanic_df)

# Convert 'Embarked' to numerical values using StringIndexer
embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex")
titanic_df = embarked_indexer.fit(titanic_df).transform(titanic_df)

# One-hot encode the 'EmbarkedIndex' column
embarked_encoder = OneHotEncoder(inputCols=["EmbarkedIndex"], outputCols=["EmbarkedVec"])
titanic_df = embarked_encoder.fit(titanic_df).transform(titanic_df)

# Define the input columns for the VectorAssembler
input_columns = ["Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkedVec"]

# Instantiate the VectorAssembler
assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

# Transform the DataFrame with the VectorAssembler
titanic_df = assembler.transform(titanic_df)

# Keep only the 'Survived' and 'features' columns for the final dataset
titanic_df = titanic_df.select(["Survived", "features"])


###########################################################################################################
################################# TRAINING & EVALUATING MODEL SPARK MLLIB #################################
###########################################################################################################

# Split the data in a train/test sample
train_data, test_data = titanic_df.randomSplit([0.8, 0.2], seed=42)

# Instantiate the classifier and set the parameters
rf_classifier = RandomForestClassifier(
    labelCol="Survived",
    featuresCol="features",
    numTrees=100,
    maxDepth=5,
    seed=42
)

# Train the classifier using the training data
rf_model = rf_classifier.fit(train_data)

# Make predictions using the testing data
predictions = rf_model.transform(test_data)

# Instantiate evaluators for each metric
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="accuracy"
)
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision"
)
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="weightedRecall"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", predictionCol="prediction", metricName="f1"
)

# Calculate the metrics using the evaluators and print the results
accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1_score))

# Saving the trained model
rf_model.save('titanic_classifier')