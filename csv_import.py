# import libraries 
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

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

