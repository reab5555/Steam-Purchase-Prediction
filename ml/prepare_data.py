# prepare_data.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer, OneHotEncoder
import argparse
import logging
from pyspark.ml import Pipeline
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_latest_sentiment_data(bucket_name):
    """Get the latest sentiment data directory from S3"""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3a.access.key", Config.AWS_ACCESS_KEY_ID) \
        .config("spark.hadoop.fs.s3a.secret.key", Config.AWS_SECRET_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.eu-central-1.amazonaws.com") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.11.901") \
        .getOrCreate()
        
    s3_path = f's3a://{bucket_name}/steam_games_reviews_s_raw/'
    df = spark.read.parquet(s3_path)
    return df

def run_data_preparation(output_path, bucket_name):
    """Run data preparation job using PySpark"""
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("DataPreparation") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
            .config("spark.hadoop.fs.s3a.access.key", Config.AWS_ACCESS_KEY_ID) \
            .config("spark.hadoop.fs.s3a.secret.key", Config.AWS_SECRET_ACCESS_KEY) \
            .config("spark.hadoop.fs.s3a.endpoint", f"s3.eu-central-1.amazonaws.com") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.11.901") \
            .getOrCreate()
    
        # Get latest sentiment data
        df = get_latest_sentiment_data(bucket_name)
        logger.info(f"Loaded data from: s3a://{bucket_name}/steam_games_reviews_s_raw/")
    
        # Define numeric columns
        numeric_columns = [
            "author_num_games_owned",
            "author_num_reviews",
            "author_playtime_forever",
            "author_playtime_last_two_weeks",
            "author_playtime_at_review",
            "comment_count",
            "positive_score",
            "negative_score"
        ]
    
        # Select relevant columns
        df = df.select(
            "game",
            "author_num_games_owned",
            "author_num_reviews",
            "author_playtime_forever",
            "author_playtime_last_two_weeks",
            "author_playtime_at_review",
            "language",
            "comment_count",
            "sentiment_analysis",
            "steam_purchase",
            "voted_up"
        )
    
        # Extract sentiment scores
        df = df.withColumn("positive_score", col("sentiment_analysis.positive_score")) \
               .withColumn("negative_score", col("sentiment_analysis.negative_score")) \
               .drop("sentiment_analysis")
    
        # Handle nulls in numeric columns
        df = df.na.fill(0, numeric_columns)
    
        # Convert numeric columns to double
        for column in numeric_columns:
            df = df.withColumn(
                column,
                when(col(column).isNull(), 0.0)
                .otherwise(col(column).cast(DoubleType()))
            )
    
        # Handle categorical columns
        categorical_columns = ["game", "language"]
    
        # Handle nulls in categorical columns
        df = df.na.fill("unknown", categorical_columns)
    
        # String Indexing
        indexers = [StringIndexer(inputCol=c, 
                                  outputCol=f"{c}_index", 
                                  handleInvalid="keep") 
                    for c in categorical_columns]
    
        # One Hot Encoding
        encoders = [OneHotEncoder(inputCol=f"{c}_index", 
                                  outputCol=f"{c}_encoded",
                                  handleInvalid="keep")
                    for c in categorical_columns]
    
        # Apply transformations
        pipeline_stages = indexers + encoders
        pipeline = Pipeline(stages=pipeline_stages)
        df = pipeline.fit(df).transform(df)
    
        # Scale numeric features
        assembler = VectorAssembler(
            inputCols=numeric_columns,
            outputCol="numeric_features",
            handleInvalid="keep"
        )
        df = assembler.transform(df)
    
        scaler = MinMaxScaler(
            inputCol="numeric_features",
            outputCol="scaled_numeric_features"
        )
        df = scaler.fit(df).transform(df)
    
        # Combine all features
        feature_cols = [f"{c}_encoded" for c in categorical_columns] + ["scaled_numeric_features"]
    
        assembler_final = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        df = assembler_final.transform(df)
    
        # Select final columns and convert target to integer
        final_df = df.select(
            "features", 
            when(col("steam_purchase").isNull(), 0)
            .otherwise(col("steam_purchase").cast("integer")).alias("label")
        )
    
        # Split the data
        train_df, validation_df = final_df.randomSplit([0.8, 0.2], seed=42)
    
        # Write training data
        train_path = f"{output_path}/train_set"
        validation_path = f"{output_path}/val_set"
    
        logger.info(f"Writing training data to: {train_path}")
        train_df.write.mode('overwrite').parquet(train_path)
    
        logger.info(f"Writing validation data to: {validation_path}")
        validation_df.write.mode('overwrite').parquet(validation_path)
    
        spark.stop()
        
        return output_path
    except Exception as e:
        logger.error(f"Data preparation failed with error: {e}")
        raise
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='S3 path to save prepared data')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    args = parser.parse_args()

    output_path = args.output
    bucket_name = args.bucket
    logger.info(f"Output path: {output_path}")
    run_data_preparation(output_path, bucket_name)

if __name__ == "__main__":
    main()
