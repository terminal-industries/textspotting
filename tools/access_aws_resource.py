import boto3
import time
import argparse
import os

from botocore.exceptions import ClientError
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')



def wait_until_instance_running(instance_id,region):
    ec2 = boto3.client('ec2', 
                       region_name=region,
                       aws_access_key_id=aws_access_key_id,
                       aws_secret_access_key=aws_secret_access_key)
    waiter = ec2.get_waiter('instance_running')
    try:
        print(f"Waiting for instance {instance_id} to enter running state...")
        waiter.wait(InstanceIds=[instance_id])
        print(f"Instance {instance_id} is now running.")
        exit(1)
    except ClientError as e:
        print(f"Failed to wait for instance state: {e}")


def get_security_group_id(ec2_client, group_name):
    # This function finds the security group ID for a given name
    try:
        response = ec2_client.describe_security_groups(GroupNames=[group_name])
        return response['SecurityGroups'][0]['GroupId']
    except ec2_client.exceptions.ClientError as e:
        print(f"Error finding security group: {e}")
        return None
def create_key_pair(ec2_client, key_name):
    # This function creates a key pair and saves the private key to a file
    try:
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        with open(f"{key_name}.pem", 'w') as file:
            file.write(key_pair['KeyMaterial'])
        print(f"Key pair created and saved as {key_name}.pem")
    except ec2_client.exceptions.ClientError as e:
        if 'InvalidKeyPair.Duplicate' in str(e):
            print("Key pair already exists.")
        else:
            raise e

def check_and_allocate_a100_instance():
    region='us-west-2'
    default_pem = 'steven-us-pem2'
    security_group_name = 'launch-wizard-45'
    instance_type = 'p3dn.24xlarge'
    key_name = default_pem
    ec2 = boto3.client('ec2',
                       region_name=region,
                       aws_access_key_id = aws_access_key_id,
                       aws_secret_access_key = aws_secret_access_key)
    create_key_pair(ec2, default_pem)

    security_group_id = get_security_group_id(ec2, security_group_name)
    if security_group_id is None:
        print("Failed to find security group.")
        return    

    while True:
        try:
            response = ec2.run_instances(
                ImageId='ami-0ca5ad49cf30fb311',
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=key_name,
                SecurityGroupIds=[security_group_id],
                Placement={
                    'AvailabilityZone': f'{region}a'
                }
            )
            instance_id = response['Instances'][0]['InstanceId']
            print(f"Successfully allocated A100 instance: {instance_id}")
            break
        except Exception as e:
            print("Failed to allocate A100 instance. Retrying in 60 seconds...")
            print(e)
            time.sleep(10)

def start_ec2_instance():
    #instance_id = 'i-0286c1afc95d3abdd'
    #region = 'us-east-2'

    instance_id = 'i-03ee1cf9ea18858c1'
    region = 'us-west-2'


    ec2 = boto3.client('ec2', 
                    region_name=region,
                    aws_access_key_id = aws_access_key_id,
                    aws_secret_access_key = aws_secret_access_key)
    while True:
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            print(f"Instance {instance_id} is starting...")
            wait_until_instance_running(instance_id,region,aws_access_key_id,aws_secret_access_key)
        except ClientError as e:
            if 'InsufficientInstanceCapacity' in str(e):
                print("Insufficient capacity. Retrying...")   
                time.sleep(5)
            else:
                print(f"An error occurred: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AWS cmds.')
    parser.add_argument('-c', '--choice', type=int, help='choice of aws', default=0)
    parser.add_argument('-i', '--key_id', type=str, help='key id of aws', default='')
    parser.add_argument('-k', '--sec_key', type=str, help='access key of aws', default='')

    args = parser.parse_args()

    if args.choice == 0: # Allocate AWS EC2 instance from AMI IDs
        check_and_allocate_a100_instance()
    elif  args.choice == 1:        
        start_ec2_instance() # start AWS EC2 instance IDs
    else:
        print(f'commnad choice {args.choice} is not found')        
