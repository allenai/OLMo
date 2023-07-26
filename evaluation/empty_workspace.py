import sys

from tango.integrations.gs.common import empty_bucket, empty_datastore

if __name__ == "__main__":
    bucket_name = sys.argv[1]
    empty_bucket(bucket_name)
    empty_datastore(bucket_name)
