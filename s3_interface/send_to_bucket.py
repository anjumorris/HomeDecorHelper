import boto3

FILENAMES = ['../cnn/weights.15-1.41.hdf5']
OUR_BUCKET = 'home-decor-bucket'

s3 = boto3.client('s3')

def make_bucket(s3=s3, bucket_name=OUR_BUCKET):
  response = s3.list_buckets()
  bucket_names = [bucket['Name'] for bucket in response['Buckets']]
  print(bucket_names)
  print(bucket_name)
  if bucket_name not in bucket_names:
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'us-west-2'})

def retrieve_from_bucket():
  s3 = boto3.resource('s3')
  s3.meta.client.download_file(OUR_BUCKET, 'weights.15-1.41.hdf5', FILENAMES[0])

if __name__ == '__main__':

  make_bucket()

  for filename in FILENAMES:
    basename = filename.split('/')[-1]
    print(basename)
    s3.upload_file(filename, OUR_BUCKET, basename)
