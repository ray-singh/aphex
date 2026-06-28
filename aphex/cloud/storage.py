"""Cloud storage backends for the aphex model registry.

Supported URI schemes
---------------------
s3://bucket/prefix     → S3Backend   (requires boto3)
gs://bucket/prefix     → GCSBackend  (requires google-cloud-storage)
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract interface for reading and writing registry artifacts."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return all keys whose path starts with *prefix* (relative to registry root)."""

    @abstractmethod
    def download(self, key: str, dest: Path) -> None:
        """Stream the object at *key* to a local file at *dest*."""

    @abstractmethod
    def upload(self, key: str, src: Path) -> None:
        """Stream a local file at *src* to the object at *key*."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists in the registry."""

    # ── convenience helpers (implemented via the abstract primitives) ──────────

    def read_text(self, key: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "content"
            self.download(key, dest)
            return dest.read_text()

    def write_text(self, key: str, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "content"
            src.write_text(text)
            self.upload(key, src)


# ── provider backends ─────────────────────────────────────────────────────────


class S3Backend(StorageBackend):
    """Amazon S3 backend. Credentials from the standard AWS environment."""

    def __init__(self, bucket: str, prefix: str) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "S3 backend requires boto3. Install with: pip install boto3"
            )
        self._s3 = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _full(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip(self, full_key: str) -> str:
        prefix_slash = f"{self._prefix}/"
        return full_key[len(prefix_slash):] if full_key.startswith(prefix_slash) else full_key

    def list(self, prefix: str) -> list[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._full(prefix)):
            for obj in page.get("Contents", []):
                keys.append(self._strip(obj["Key"]))
        return keys

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            self._s3.download_fileobj(self._bucket, self._full(key), fh)

    def upload(self, key: str, src: Path) -> None:
        with src.open("rb") as fh:
            self._s3.upload_fileobj(fh, self._bucket, self._full(key))

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._full(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend. Credentials from Application Default Credentials."""

    def __init__(self, bucket: str, prefix: str) -> None:
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "GCS backend requires google-cloud-storage. "
                "Install with: pip install google-cloud-storage"
            )
        try:
            self._client = gcs.Client()
        except Exception as exc:
            raise RuntimeError(
                "GCS backend: could not initialise client. Ensure Application "
                "Default Credentials are configured (`gcloud auth application-"
                "default login` or GOOGLE_APPLICATION_CREDENTIALS)."
            ) from exc
        self._bucket_obj = self._client.bucket(bucket)
        self._prefix = prefix.rstrip("/")

    def _full(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _strip(self, full_key: str) -> str:
        prefix_slash = f"{self._prefix}/"
        return full_key[len(prefix_slash):] if full_key.startswith(prefix_slash) else full_key

    def list(self, prefix: str) -> list[str]:
        blobs = self._client.list_blobs(self._bucket_obj, prefix=self._full(prefix))
        return [self._strip(b.name) for b in blobs]

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self._bucket_obj.blob(self._full(key))
        blob.download_to_filename(str(dest))

    def upload(self, key: str, src: Path) -> None:
        blob = self._bucket_obj.blob(self._full(key))
        blob.upload_from_filename(str(src))

    def exists(self, key: str) -> bool:
        return self._bucket_obj.blob(self._full(key)).exists()


class MemoryBackend(StorageBackend):
    """In-memory backend for tests. Not for production use."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def list(self, prefix: str) -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]

    def download(self, key: str, dest: Path) -> None:
        if key not in self._store:
            raise FileNotFoundError(f"Key not found in registry: {key!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._store[key])

    def upload(self, key: str, src: Path) -> None:
        self._store[key] = src.read_bytes()

    def exists(self, key: str) -> bool:
        return key in self._store


# ── factory ───────────────────────────────────────────────────────────────────


def get_backend(uri: str) -> StorageBackend:
    """Parse a registry URI and return the appropriate StorageBackend.

    Supported schemes: s3://, gs://
    """
    if uri.startswith("s3://"):
        rest = uri[5:]
        bucket, _, prefix = rest.partition("/")
        return S3Backend(bucket, prefix)
    if uri.startswith("gs://"):
        rest = uri[5:]
        bucket, _, prefix = rest.partition("/")
        return GCSBackend(bucket, prefix)
    raise ValueError(
        f"Unsupported registry URI: {uri!r}. "
        "Expected s3://bucket/prefix or gs://bucket/prefix."
    )
