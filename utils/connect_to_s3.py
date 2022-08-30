import boto3
import config

s3 = boto3.resource(
    service_name=config.service_name,
    region_name=config.region_name,
    aws_access_key_id=config.aws_access_key_id,
    aws_secret_access_key=config.aws_secret_access_key
)
