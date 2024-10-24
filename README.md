# Steam Game Reviews Sentiment Analysis & Prediction Pipeline

This project is designed to analyze and predict sentiments from Steam game reviews using AWS infrastructure, Apache Airflow for orchestration, and a combination of tools including Apache Spark, SageMaker, and XGBoost.

## Project Overview

The goal of the project is to build a sentiment analysis and prediction pipeline for over 2.4M Steam game reviews. The process begins by extracting raw review data stored in MongoDB, performing sentiment analysis using Hugging Face models, and finally training a machine learning model (XGBoost) on the pre-processed data to predict user behaviors or sentiments.

![Project Diagram](./ERDs - Steam.png)

### Workflow:

1. **Data Ingestion (MongoDB -> Parquet)**:
   - Steam game reviews are fetched from a MongoDB database.
   - The data is processed into Parquet format using Apache Airflow, and stored in AWS S3 as the data lake.

2. **Sentiment Analysis**:
   - A Hugging Face model deployed on AWS SageMaker performs sentiment analysis on the review texts.
   - AWS Comprehend is also integrated to extract insights, such as positive and negative sentiment scores.

3. **Data Preprocessing**:
   - Reviews are sampled using AWS EMR (Elastic MapReduce) and Apache Spark.
   - The preprocessed data includes scaling and encoding features to prepare for model training.

4. **Model Training & Evaluation**:
   - XGBoost, a scalable and high-performance machine learning library, is used for training on the pre-processed data in SageMaker.
   - The model is then evaluated using metrics such as AUC (Area Under the Curve).

## Key Components

### 1. **Data Preparation**
   - Script: [`prepare_data.py`](./prepare_data.py)
   - This script performs data cleaning and transformation using PySpark. It extracts the relevant features, scales numeric columns, and encodes categorical features.
   - Processed data is split into training and validation sets and stored back in S3.

### 2. **Sentiment Analysis**
   - Script: [`sentiment_analysis.py`](./sentiment_analysis.py)
   - A pre-trained Hugging Face model (`distilbert-base-uncased-finetuned-sst-2-english`) deployed on AWS SageMaker is used to analyze the sentiment of game reviews.
   - The results (positive/negative scores) are written back to S3 in Parquet format.

### 3. **Model Training**
   - Script: [`train.py`](./train.py)
   - A gradient-boosted tree classifier (GBTClassifier) is trained using the pre-processed features.
   - Evaluation metrics (such as AUC) are logged, and the trained model is saved in S3 for future use.

### 4. **EMR Cluster Management**
   - Script: [`emr_cluster_manager.py`](./emr_cluster_manager.py)
   - This script manages the lifecycle of an AWS EMR cluster, which is used for distributed computing with Spark. It can create, manage, and terminate EMR clusters as needed.

### 5. **Workflow Orchestration**
   - Script: [`main.py`](./main.py)
   - This script orchestrates the entire workflow, using the EMR cluster to prepare data and train the model. It also handles error management and logs the progress of each step.

## Technology Stack

- **Orchestration**: Apache Airflow
- **Data Storage**: MongoDB, AWS S3
- **Data Processing**: Apache Spark, AWS EMR
- **Machine Learning**: Hugging Face, AWS SageMaker, XGBoost
- **Programming Languages**: Python (PySpark, boto3 for AWS SDK)
- **Logging**: Python Logging module

## Setup & Usage

### Requirements

- AWS Account with access to S3, SageMaker, and EMR.
- MongoDB instance with Steam reviews dataset.
- Python environment with the following packages:
  - PySpark
  - boto3
  - sagemaker
  - HuggingFace Transformers

### Steps

1. **MongoDB to Parquet**: Set up an Apache Airflow DAG to extract data from MongoDB and load it into S3 as Parquet.
2. **Run Sentiment Analysis**: Deploy the sentiment analysis model using SageMaker and process the reviews.
3. **Data Preprocessing**: Run the `prepare_data.py` script to prepare the data for training.
4. **Model Training**: Execute the `train.py` script to train the XGBoost model using Spark on AWS EMR.
5. **Orchestrate the Workflow**: Use `main.py` to orchestrate the entire process from data preparation to model training and evaluation.

### Sample Commands

To run the sentiment analysis:

```bash
python sentiment_analysis.py
