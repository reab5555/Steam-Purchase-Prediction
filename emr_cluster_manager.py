import boto3
import time
import logging
from botocore.exceptions import ClientError
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EMRClusterManager:
    def __init__(self):
        self.emr_client = boto3.client('emr', region_name=Config.AWS_REGION)
        self.cluster_id = None

    def create_cluster(self):
        logger.info("Creating EMR cluster...")
        try:
            response = self.emr_client.run_job_flow(
                Name='SteamReviewsCluster',
                LogUri=f's3://{Config.S3_BUCKET}/emr-logs/',
                ReleaseLabel='emr-7.3.0',
                Applications=[
                    {'Name': 'Hadoop'},
                    {'Name': 'Spark'}
                ],
                Instances={
                    'InstanceGroups': [
                        {
                            'Name': 'Master nodes',
                            'Market': 'ON_DEMAND',
                            'InstanceRole': 'MASTER',
                            'InstanceType': Config.SPARK_INSTANCE_TYPE,
                            'InstanceCount': 1,
                        },
                        {
                            'Name': 'Worker nodes',
                            'Market': 'ON_DEMAND',
                            'InstanceRole': 'CORE',
                            'InstanceType': Config.SPARK_INSTANCE_TYPE,
                            'InstanceCount': Config.SPARK_INSTANCE_COUNT - 1,
                        }
                    ],
                    'Ec2KeyName': Config.EC2_KEY_NAME,
                    'KeepJobFlowAliveWhenNoSteps': True,
                    'TerminationProtected': False,
                    'Ec2SubnetId': Config.EC2_SUBNET_ID,
                },
                BootstrapActions=[
                    {
                        'Name': 'Install dependencies',
                        'ScriptBootstrapAction': {
                        'Path': f's3://{Config.S3_BUCKET}/train/scripts/install_dependencies.sh'                        }
                    }
                ],
                JobFlowRole=Config.EMR_EC2_ROLE,
                ServiceRole=Config.EMR_SERVICE_ROLE,
                VisibleToAllUsers=True,
                Tags=[
                    {
                        'Key': 'Project',
                        'Value': 'SteamReviews'
                    }
                ]
            )
            self.cluster_id = response['JobFlowId']
            logger.info(f"EMR cluster created with ID: {self.cluster_id}")
            self.wait_for_cluster()
            return self.cluster_id
        except ClientError as e:
            logger.error(f"Failed to create EMR cluster: {e}")
            raise

    def wait_for_cluster(self):
        logger.info("Waiting for EMR cluster to be ready...")
        waiter = self.emr_client.get_waiter('cluster_running')
        waiter.wait(ClusterId=self.cluster_id)
        logger.info("EMR cluster is ready.")

    def terminate_cluster(self):
        logger.info(f"Terminating EMR cluster with ID: {self.cluster_id}")
        try:
            self.emr_client.terminate_job_flows(
                JobFlowIds=[self.cluster_id]
            )
            logger.info("EMR cluster terminated.")
        except ClientError as e:
            logger.error(f"Failed to terminate EMR cluster: {e}")
            raise
