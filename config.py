import os

class Config:
    # AWS Credentials (loaded from environment variables)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    # AWS Configuration
    AWS_REGION = 'AWS_REGION'
    S3_BUCKET = "S3_BUCKET"

    # EMR IAM Roles
    EMR_EC2_ROLE = "EMR_EC2_DefaultRole"
    EMR_SERVICE_ROLE = "EMR_DefaultRole"
    EC2_KEY_NAME = "pemkey"
    EC2_SUBNET_ID = "EC2_SUBNET_ID"
    
    # Spark Configuration for Data Processing
    SPARK_INSTANCE_TYPE = "m5.xlarge"
    SPARK_INSTANCE_COUNT = 3  # 1 master + 2 core nodes