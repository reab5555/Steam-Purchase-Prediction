import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import logging
import json
from config import Config
import os
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_endpoint_name():
    """Generate a unique endpoint name"""
    return "huggingface-sentiment-endpoint"

def deploy_model():
    """Deploy the HuggingFace model and return the endpoint name"""
    logger.info("Deploying HuggingFace model...")
    
    endpoint_name = get_endpoint_name()
    sagemaker_client = boto3.client('sagemaker')
    
    # Check and clean up existing endpoint and config
    try:
        # Check if endpoint exists and wait for it to be ready before deleting
        logger.info(f"Checking for existing endpoint: {endpoint_name}")
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            if response['EndpointStatus'] in ['Creating', 'Updating']:
                logger.info(f"Endpoint is {response['EndpointStatus']}. Waiting for it to be ready...")
                waiter = sagemaker_client.get_waiter('endpoint_in_service')
                waiter.wait(EndpointName=endpoint_name)
            
            logger.info(f"Found existing endpoint. Deleting endpoint: {endpoint_name}")
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            # Wait for endpoint deletion
            waiter = sagemaker_client.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted successfully: {endpoint_name}")
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                logger.info("No existing endpoint found")
            else:
                raise
        
        # Check and delete endpoint config if exists
        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info(f"Found existing endpoint config. Deleting config: {endpoint_name}")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info(f"Endpoint config deleted successfully: {endpoint_name}")
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint configuration" in str(e):
                logger.info("No existing endpoint config found")
            else:
                raise
        
        # Create new model and endpoint
        logger.info("Creating new endpoint...")
        hub = {
            'HF_MODEL_ID': 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
            'HF_TASK': 'text-classification'
        }

        huggingface_model = HuggingFaceModel(
            transformers_version='4.37.0',
            pytorch_version='2.1.0',
            py_version='py310',
            env=hub,
            role=Config.SAGEMAKER_ROLE # Your Sagemaker Role
        )
        
        # Deploy with memory optimized instance
        huggingface_model.deploy(
            initial_instance_count=1,
            instance_type='ml.r5.4xlarge',
            endpoint_name=endpoint_name
        )
        
        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise

def create_spark_session():
    """Create Spark session for reading Parquet"""
    logger.info("Initializing Spark session...")
    
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3a.access.key", Config.AWS_ACCESS_KEY_ID) \
        .config("spark.hadoop.fs.s3a.secret.key", Config.AWS_SECRET_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{Config.AWS_REGION}.amazonaws.com") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.11.901") \
        .getOrCreate()
        
    logger.info("Spark session initialized successfully")
    return spark

def create_analyze_review_udf(endpoint_name, region):
    """Create UDF for sentiment analysis with endpoint configuration"""
    def truncate_text(text, max_chars=500):
        if not text:
            return text
        return text[:max_chars]
    
    def analyze_review(review):
        if not review:
            return {
                "label": "NEUTRAL",
                "score": 0,
                "positive_score": 0,
                "negative_score": 0
            }
    
        try:
            import boto3
            runtime = boto3.client('sagemaker-runtime', region_name=region)
    
            cleaned_review = review.replace('"', '\\"').replace('\n', ' ')
            truncated_review = truncate_text(cleaned_review)
    
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({
                    "inputs": truncated_review,
                    "parameters": {
                        "truncation": True,
                        "max_length": 256
                    }
                })
            )
    
            result = json.loads(response['Body'].read().decode())
    
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                result_item = result[0]
                label = result_item.get('label', 'NEUTRAL')
                score = float(result_item.get('score', 0.5))
    
                positive_score = score if label == 'POSITIVE' else 1 - score
                negative_score = score if label == 'NEGATIVE' else 1 - score
    
                return {
                    "label": label,
                    "score": score,
                    "positive_score": positive_score,
                    "negative_score": negative_score
                }
            else:
                logger.warning(f"Unexpected result format: {result}")
                return {
                    "label": "ERROR",
                    "score": 0.0,
                    "positive_score": 0.0,
                    "negative_score": 0.0
                }
    
        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}")
            return {
                "label": "ERROR",
                "score": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0
            }
    
    return udf(analyze_review, StructType([
        StructField("label", StringType(), True),
        StructField("score", FloatType(), True),
        StructField("positive_score", FloatType(), True),
        StructField("negative_score", FloatType(), True)
    ]))

def process_parquet_data():
    """Process Parquet data and add sentiment analysis"""
    endpoint_name = None
    spark = None
    
    try:
        logger.info("Starting sentiment analysis processing...")
        
        # Deploy the model first
        endpoint_name = deploy_model()
        
        # Create Spark session
        spark = create_spark_session()
        
        # Generate unique timestamp for file names
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info("Reading data from Parquet file...")
        input_path = "s3a://test24214/steam_games_reviews_source/steam_reviews_sample.parquet"
        logger.info(f"Reading from: {input_path}")
        df = spark.read.parquet(input_path)
        
        logger.info("Applying sentiment analysis...")
        sentiment_udf = create_analyze_review_udf(endpoint_name, Config.AWS_REGION)
        
        # Process in batches
        df_with_sentiment = df.withColumn(
            "sentiment_analysis",
            sentiment_udf(col("review"))
        ).repartition(10)
        
        # Write to Parquet format with unique ID
        output_path = f"s3a://test24214/steam_games_reviews_s_raw/steam_reviews_sentiment_{timestamp}"
        logger.info(f"Writing results to Parquet format at: {output_path}")
        
        # Write as Parquet with compression
        df_with_sentiment.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)
        
        logger.info("Sentiment analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        raise
    finally:
        # Cleanup
        if endpoint_name:
            logger.info("Cleaning up SageMaker endpoint...")
            try:
                sagemaker_client = boto3.client('sagemaker')
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            except Exception as e:
                logger.error(f"Error deleting endpoint: {str(e)}")
        
        if spark:
            logger.info("Stopping Spark session...")
            try:
                spark.stop()
            except Exception as e:
                logger.error(f"Error stopping Spark: {str(e)}")

def main():
    try:
        process_parquet_data()
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()