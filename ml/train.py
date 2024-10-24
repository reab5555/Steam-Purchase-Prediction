from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # Paths to training and validation data
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation data')
    parser.add_argument('--model_output', type=str, default='xgb_spark_model', help='Path to save the trained model')

    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SparkModelTraining") \
        .getOrCreate()

    # Load training and validation data
    logger.info(f"Loading training data from {args.train}")
    train_df = spark.read.parquet(args.train)
    logger.info(f"Loading validation data from {args.validation}")
    validation_df = spark.read.parquet(args.validation)

    # Assemble features into a vector (excluding the label column)
    feature_columns = [col for col in train_df.columns if col != 'label']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    # Define the classifier (using Gradient Boosted Trees as an example)
    classifier = GBTClassifier(
        featuresCol='features',
        labelCol='label',
        maxIter=50
    )

    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, classifier])

    # Train the model
    logger.info("Starting training...")
    model = pipeline.fit(train_df)

    # Evaluate the model
    logger.info("Evaluating model...")
    predictions = model.transform(validation_df)
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
    auc = evaluator.evaluate(predictions)
    logger.info(f"Validation AUC: {auc}")

    # Save the model
    model_output_path = args.model_output
    logger.info(f"Saving model to {model_output_path}")
    model.write().overwrite().save(model_output_path)

    spark.stop()

if __name__ == '__main__':
    main()
