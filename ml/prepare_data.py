from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, count, sum
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import argparse
import logging
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_latest_sentiment_data(bucket_name):
    """Get the latest sentiment data directory from S3"""
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
    
        # Aggregate review patterns and sentiments for each game
        game_aggregates = df.groupBy("game").agg(
            avg("positive_score").alias("avg_positive_score"),
            avg("negative_score").alias("avg_negative_score"),
            count("*").alias("total_reviews"),
            sum(when(col("voted_up") == True, 1).otherwise(0)).alias("total_positive_reviews"),
            sum(when(col("voted_up") == False, 1).otherwise(0)).alias("total_negative_reviews")
        )

        # Handle nulls in aggregated columns
        game_aggregates = game_aggregates.na.fill(0, [
            "avg_positive_score", "avg_negative_score", "total_reviews", "total_positive_reviews", "total_negative_reviews"
        ])

        # Encoding categorical features (e.g., game)
        indexer = StringIndexer(inputCol="game", outputCol="game_index", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol="game_index", outputCol="game_encoded")

        # Assemble features
        feature_cols = ["avg_positive_score", "avg_negative_score", "total_reviews", 
                        "total_positive_reviews", "total_negative_reviews", "game_encoded"]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="numeric_features")

        # Scale the numeric features
        scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")

        # Final assembler to combine features
        final_assembler = VectorAssembler(inputCols=["scaled_numeric_features"], outputCol="features")

        # Define the pipeline stages
        pipeline_stages = [indexer, encoder, assembler, scaler, final_assembler]
        pipeline = Pipeline(stages=pipeline_stages)

        # Apply the transformations
        transformed_df = pipeline.fit(game_aggregates).transform(game_aggregates)
    
        # Select final columns and convert target to integer
        final_df = transformed_df.select("features", 
                                         when(col("steam_purchase").isNull(), 0)
                                         .otherwise(col("steam_purchase").cast("integer")).alias("label"))

        # Split the data into training and validation sets
        train_df, validation_df = final_df.randomSplit([0.8, 0.2], seed=42)

        # Write training and validation data
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
