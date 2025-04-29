import boto3

def get_npy_files(bucket_name, prefix):
    s3 = boto3.client('s3')
    npy_files = []

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.npy'):
                    npy_files.append(obj['Key'])

    return npy_files

# Example usage
bucket_name = 'ai2-llm'
prefix = 'preprocessed/dclm/samples/rewrite-dclm-ft-delta-bottom-20pctl/allenai/dolma2-tokenizer/'


npy_files = get_npy_files(bucket_name, prefix)
for file in npy_files:
    print(f"- s3://{bucket_name}/{file}")
