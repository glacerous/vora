import os
import time
import boto3
from botocore.config import Config

def upload_splat(file_path: str, tree_code: str, custom_timestamp: int = None) -> str:
    """
    Uploads a tree scan file (.ply or .splat) to Cloudflare R2 bucket.
    Reads credentials from environment variables.
    Returns the public custom URL (if configured) or a 7-day presigned URL.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    
    if not all([account_id, access_key, secret_key, bucket_name]):
        raise ValueError(
            "Missing R2 configuration. Please check CLOUDFLARE_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME in environment."
        )
        
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    # Configure client to connect to R2
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto"
    )
    
    file_name = os.path.basename(file_path)
    # Store scans in a unique tree-specific prefix with timestamp to avoid name collisions
    ts = custom_timestamp if custom_timestamp is not None else int(time.time())
    object_key = f"tree_scans/{tree_code}/{ts}_{file_name}"
    
    content_type = "application/octet-stream"
    if file_name.endswith(".ply"):
        content_type = "application/x-ply"
    elif file_name.endswith(".splat"):
        content_type = "application/octet-stream"
        
    s3.upload_file(
        file_path,
        bucket_name,
        object_key,
        ExtraArgs={"ContentType": content_type}
    )
    
    # If a public custom domain is configured, use it to build a permanent public link
    public_url_prefix = os.environ.get("R2_PUBLIC_URL_PREFIX")
    if public_url_prefix:
        public_url_prefix = public_url_prefix.rstrip("/")
        return f"{public_url_prefix}/{object_key}"
        
    # Default to generating a presigned URL valid for the max limit of 7 days (604800 seconds)
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=604800
    )
    return presigned_url

def upload_thumbnail(file_path: str, tree_code: str) -> str:
    """
    Uploads a representative frame image to Cloudflare R2 bucket as a thumbnail.
    Reads credentials from environment variables.
    Returns the public custom URL (if configured) or a 7-day presigned URL.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    
    if not all([account_id, access_key, secret_key, bucket_name]):
        raise ValueError(
            "Missing R2 configuration. Please check CLOUDFLARE_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME in environment."
        )
        
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto"
    )
    
    file_name = os.path.basename(file_path)
    # Store thumbnails in separate path: thumbnails/{tree_code}/{int(time.time())}_{filename}.jpg
    object_key = f"thumbnails/{tree_code}/{int(time.time())}_{file_name}"
    
    s3.upload_file(
        file_path,
        bucket_name,
        object_key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )
    
    public_url_prefix = os.environ.get("R2_PUBLIC_URL_PREFIX")
    if public_url_prefix:
        public_url_prefix = public_url_prefix.rstrip("/")
        return f"{public_url_prefix}/{object_key}"
        
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=604800
    )
    return presigned_url

