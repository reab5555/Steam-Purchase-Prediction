import logging
import subprocess
from datetime import datetime
from emr_cluster_manager import EMRClusterManager
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.cluster_manager = EMRClusterManager()
        self.cluster_id = None

    def clean_and_prepare_data(self):
        """Clean and prepare data using Spark on EMR"""
        try:
            logger.info("Starting data preparation on EMR...")
            output_data_s3_path = f's3://{Config.S3_BUCKET}/train/prepared_data'
            
            # Path to prepare_data.py in S3
            script_s3_path = f's3://{Config.S3_BUCKET}/train/scripts/prepare_data.py'
    
            # Build the EMR step to run prepare_data.py
            step_args = [
                'spark-submit',
                '--deploy-mode', 'cluster',
                '--master', 'yarn',
                script_s3_path,
                '--output', output_data_s3_path,
                '--bucket', Config.S3_BUCKET
            ]
            
            response = self.cluster_manager.emr_client.add_job_flow_steps(
                JobFlowId=self.cluster_id,
                Steps=[
                    {
                        'Name': 'PrepareData',
                        'ActionOnFailure': 'CONTINUE',
                        'HadoopJarStep': {
                            'Jar': 'command-runner.jar',
                            'Args': step_args
                        }
                    }
                ]
            )
            step_id = response['StepIds'][0]
            logger.info(f"Data preparation step submitted with ID: {step_id}")
            self.wait_for_step_completion(step_id)
            logger.info(f"Data preparation completed and saved to {output_data_s3_path}")
            return output_data_s3_path
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def run_training(self, data_path):
        """Run the training using Spark and XGBoost on EMR"""
        try:
            logger.info(f"Starting training with data from: {data_path}")
            train_data_path = f'{data_path}/train_set'
            validation_data_path = f'{data_path}/val_set'
            model_output_path = f's3://{Config.S3_BUCKET}/train/models/xgb_spark_model_{self.timestamp}'

            # Path to training script in S3
            script_s3_path = f's3://{Config.S3_BUCKET}/train/scripts/train.py'

            # Build the EMR step to run train.py
            step_args = [
                'spark-submit',
                '--deploy-mode', 'cluster',
                '--master', 'yarn',
                script_s3_path,
                '--train', train_data_path,
                '--validation', validation_data_path,
                '--model_output', model_output_path
            ]

            response = self.cluster_manager.emr_client.add_job_flow_steps(
                JobFlowId=self.cluster_id,
                Steps=[
                    {
                        'Name': 'TrainModel',
                        'ActionOnFailure': 'CONTINUE',
                        'HadoopJarStep': {
                            'Jar': 'command-runner.jar',
                            'Args': step_args
                        }
                    }
                ]
            )
            step_id = response['StepIds'][0]
            logger.info(f"Training step submitted with ID: {step_id}")
            self.wait_for_step_completion(step_id)
            logger.info(f"Training completed. Model saved to {model_output_path}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def wait_for_step_completion(self, step_id):
        logger.info(f"Waiting for step {step_id} to complete...")
        waiter = self.cluster_manager.emr_client.get_waiter('step_complete')
        waiter.wait(
            ClusterId=self.cluster_id,
            StepId=step_id,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 240  # Increased MaxAttempts for longer steps
            }
        )
        logger.info(f"Step {step_id} completed.")

    def run_workflow(self):
        """Execute complete workflow"""
        try:
            # Step 0: Create EMR Cluster
            logger.info("Step 0: Creating EMR cluster...")
            self.cluster_id = self.cluster_manager.create_cluster()
            logger.info("Step 0: EMR cluster created successfully")

            # Step 1: Clean and Prepare Data
            logger.info("Step 1: Starting data preparation workflow...")
            data_path = self.clean_and_prepare_data()
            logger.info("Step 1: Data preparation completed successfully")

            # Step 2: Run Training
            logger.info("Step 2: Starting training workflow...")
            self.run_training(data_path)
            logger.info("Step 2: Training completed successfully")

            logger.info(f"""
            Workflow completed successfully!
            Timestamp: {self.timestamp}
            Data path: {data_path}
            """)

            return True

        except Exception as e:
            logger.error(f"Workflow failed with error: {str(e)}")
            return False

        finally:
            # Step 3: Terminate EMR Cluster
            if self.cluster_id:
                logger.info("Step 3: Terminating EMR cluster...")
                self.cluster_manager.terminate_cluster()

def main():
    try:
        logger.info("Initializing workflow orchestrator...")
        orchestrator = WorkflowOrchestrator()

        logger.info("Starting workflow execution...")
        success = orchestrator.run_workflow()

        if not success:
            logger.error("Workflow execution failed")
            raise Exception("Workflow failed")

        logger.info("Workflow execution completed successfully")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
