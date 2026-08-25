import logging
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from google.api_core.exceptions import NotFound
from google.cloud import storage


logger = logging.getLogger(__name__)


class GCSClient:
    def __init__(self, bucket_name: str = "", project_id: str = "video-backend-dev"):
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id) if project_id else storage.Client()

    def download_file(self, gcs_uri: str, destination_path: str) -> None:
        if gcs_uri.startswith("gs://"):
            path_without_scheme = gcs_uri[5:]
            bucket_name, object_name = path_without_scheme.split("/", 1)
        else:
            if not self.bucket_name:
                raise ValueError("Plain blob download requires a default bucket_name.")
            bucket_name = self.bucket_name
            object_name = gcs_uri

        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(str(destination))


def safe_download_file(gcs_client: "GCSClient", gcs_uri: str, destination_path: str) -> Tuple[bool, str]:
    if not gcs_uri:
        return False, ""
    try:
        gcs_client.download_file(gcs_uri, destination_path)
        return True, ""
    except NotFound:
        return False, "No file found"


def fetch_segment_insights(segment_insight_id: str, auth_token: str, api_base_url: str) -> Dict:
    url = f"{api_base_url.rstrip('/')}/api/v1/insights/segment-insights/{segment_insight_id}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_segment_context(
    video_id: str,
    segment_id: str,
    auth_token: str,
    api_base_url: str,
) -> Dict:
    url = (
        f"{api_base_url.rstrip('/')}/api/v1/videos/segments/"
        f"{video_id}/{segment_id}/context?include_file_path=true"
    )
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_segment_insights_with_tokens(
    segment_insight_id: str,
    auth_tokens: list[str],
    api_base_url: str,
) -> Dict:
    last_error: Optional[Exception] = None
    for token in auth_tokens:
        try:
            return fetch_segment_insights(segment_insight_id, token, api_base_url)
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError("No auth tokens provided.")


def fetch_segment_context_with_tokens(
    video_id: str,
    segment_id: str,
    auth_tokens: list[str],
    api_base_url: str,
) -> Dict:
    last_error: Optional[Exception] = None
    for token in auth_tokens:
        try:
            return fetch_segment_context(video_id, segment_id, token, api_base_url)
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError("No auth tokens provided.")


def find_release_artifact(artifacts: Dict) -> Tuple[Optional[str], Optional[str]]:
    release_key_candidates = [
        "release_point_csv",
        "release_points_csv",
        "release_point",
        "ball_release_csv",
    ]
    for key in release_key_candidates:
        if artifacts.get(key):
            return key, artifacts[key]

    for key, value in artifacts.items():
        if "release" in key.lower() and isinstance(value, str):
            return key, value
    return None, None


def build_keypoints_uri_from_context(
    context_payload: Dict,
    segment_id: str,
    segment_insight_id: str,
) -> Optional[str]:
    video_file_path = (
        context_payload.get("video", {}).get("file_path")
        or context_payload.get("segment", {}).get("file_path")
        or ""
    )
    if not isinstance(video_file_path, str) or not video_file_path.startswith("gs://"):
        return None

    path_without_scheme = video_file_path[5:]
    bucket_name, object_name = path_without_scheme.split("/", 1)
    object_parts = object_name.split("/")
    if len(object_parts) < 5:
        return None

    base_parts = object_parts[:-1]
    video_id = context_payload.get("video", {}).get("id") or context_payload.get("segment", {}).get("video_id")
    if video_id:
        # Existing successful downloads use .../<video_id>/<video_id>/<segment_id>/<segment_insight_id>/bowler_keypoints.csv
        base_parts.append(str(video_id))

    keypoint_parts = base_parts + [str(segment_id), str(segment_insight_id), "bowler_keypoints.csv"]
    return f"gs://{bucket_name}/{'/'.join(keypoint_parts)}"


def download_artifacts(
    manifest_csv: str,
    output_dir: str,
    api_base_url: str,
    auth_token: str = "",
    auth_tokens: Optional[list[str]] = None,
    bucket_name: str = "",
    project_id: str = "video-backend-dev",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if limit is not None:
        df = df.head(limit)

    output_root = Path(output_dir)
    keypoints_dir = output_root / "keypoints"
    release_dir = output_root / "release_points"
    videos_dir = output_root / "videos"
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    gcs_client = GCSClient(bucket_name=bucket_name, project_id=project_id)
    token_list = [token for token in (auth_tokens or [auth_token]) if token]
    results = []

    for index, row in df.iterrows():
        segment_id = row.get("segment_id") or row.get("segmentId")
        segment_insight_id = row.get("segment_insight_id") or row.get("segmentInsightId")
        if not segment_id or not segment_insight_id:
            raise ValueError("Manifest must contain segment_id and segment_insight_id columns.")

        video_id = row.get("Video ID", row.get("video_id", "unknown_video"))
        ball = row.get("Ball", row.get("ball", index + 1))
        logger.info("Processing %s_%s (%s)", video_id, ball, segment_id)

        keypoints_gcs_uri = ""
        release_artifact_key = ""
        release_gcs_uri = ""
        video_gcs_uri = ""

        try:
            response = fetch_segment_insights_with_tokens(str(segment_insight_id), token_list, api_base_url)
            data = response.get("data", {})
            artifacts = data.get("results", {}).get("artifacts", {})
            keypoints_gcs_uri = artifacts.get("pose_data_csv") or ""
            release_artifact_key, release_gcs_uri = find_release_artifact(artifacts)
            release_artifact_key = release_artifact_key or ""
            release_gcs_uri = release_gcs_uri or ""
        except Exception:
            pass

        context_payload = fetch_segment_context_with_tokens(str(video_id), str(segment_id), token_list, api_base_url)
        video_gcs_uri = context_payload.get("segment", {}).get("file_path") or ""
        if not keypoints_gcs_uri:
            keypoints_gcs_uri = (
                build_keypoints_uri_from_context(context_payload, str(segment_id), str(segment_insight_id)) or ""
            )

        keypoints_path = keypoints_dir / f"{segment_id}_keypoints.csv"
        release_path = release_dir / f"{segment_id}_release.csv"
        video_path = videos_dir / f"{video_id}_ball{ball}.mp4"

        keypoints_downloaded, keypoints_error = safe_download_file(
            gcs_client, keypoints_gcs_uri, str(keypoints_path)
        )
        release_downloaded, release_error = safe_download_file(
            gcs_client, release_gcs_uri, str(release_path)
        )
        video_downloaded, video_error = safe_download_file(
            gcs_client, video_gcs_uri, str(video_path)
        )

        if keypoints_gcs_uri and not keypoints_downloaded:
            logger.warning("No bowler keypoints found for segment %s", segment_id)

        results.append(
            {
                "video_id": video_id,
                "ball": ball,
                "segment_id": segment_id,
                "segment_insight_id": segment_insight_id,
                "keypoints_gcs_uri": keypoints_gcs_uri or "",
                "release_artifact_key": release_artifact_key or "",
                "release_gcs_uri": release_gcs_uri or "",
                "keypoints_path": str(keypoints_path) if keypoints_downloaded else "",
                "release_point_path": str(release_path) if release_downloaded else "",
                "video_gcs_uri": video_gcs_uri,
                "video_path": str(video_path) if video_downloaded else "",
                "keypoints_status": "downloaded" if keypoints_downloaded else "missing",
                "release_status": "downloaded" if release_downloaded else ("missing" if release_gcs_uri else ""),
                "video_status": "downloaded" if video_downloaded else ("missing" if video_gcs_uri else ""),
                "keypoints_error": keypoints_error,
                "release_error": release_error,
                "video_error": video_error,
            }
        )

    summary_df = pd.DataFrame(results)
    summary_path = output_root / "download_summary.csv"
    summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return summary_df


def fetch_single_segment_artifacts(
    video_id: str,
    segment_id: str,
    segment_insight_id: str,
    output_dir: str,
    api_base_url: str,
    auth_token: str = "",
    auth_tokens: Optional[list[str]] = None,
    bucket_name: str = "",
    project_id: str = "video-backend-dev",
) -> Dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    keypoints_dir = output_root / "keypoints"
    release_dir = output_root / "release_points"
    videos_dir = output_root / "videos"
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    gcs_client = GCSClient(bucket_name=bucket_name, project_id=project_id)
    token_list = [token for token in (auth_tokens or [auth_token]) if token]
    keypoints_gcs_uri = ""
    release_artifact_key = ""
    release_gcs_uri = ""
    video_gcs_uri = ""
    try:
        response = fetch_segment_insights_with_tokens(segment_insight_id, token_list, api_base_url)
        data = response.get("data", {})
        artifacts = data.get("results", {}).get("artifacts", {})
        keypoints_gcs_uri = artifacts.get("pose_data_csv") or ""
        release_artifact_key, release_gcs_uri = find_release_artifact(artifacts)
        release_artifact_key = release_artifact_key or ""
        release_gcs_uri = release_gcs_uri or ""
    except Exception:
        pass
    context_payload = fetch_segment_context_with_tokens(str(video_id), str(segment_id), token_list, api_base_url)
    video_gcs_uri = context_payload.get("segment", {}).get("file_path") or ""
    if not keypoints_gcs_uri:
        keypoints_gcs_uri = build_keypoints_uri_from_context(
            context_payload,
            str(segment_id),
            str(segment_insight_id),
        ) or ""

    keypoints_path = keypoints_dir / f"{segment_id}_keypoints.csv"
    release_path = release_dir / f"{segment_id}_release.csv"
    video_path = videos_dir / f"{video_id}_{segment_id}.mp4"
    keypoints_downloaded, keypoints_error = safe_download_file(
        gcs_client, keypoints_gcs_uri, str(keypoints_path)
    )
    release_downloaded, release_error = safe_download_file(
        gcs_client, release_gcs_uri, str(release_path)
    )
    video_downloaded, video_error = safe_download_file(
        gcs_client, video_gcs_uri, str(video_path)
    )

    return {
        "segment_id": segment_id,
        "segment_insight_id": segment_insight_id,
        "keypoints_gcs_uri": keypoints_gcs_uri,
        "release_artifact_key": release_artifact_key or "",
        "release_gcs_uri": release_gcs_uri,
        "keypoints_path": str(keypoints_path) if keypoints_downloaded else "",
        "release_point_path": str(release_path) if release_downloaded else "",
        "video_gcs_uri": video_gcs_uri,
        "video_path": str(video_path) if video_downloaded else "",
        "keypoints_status": "downloaded" if keypoints_downloaded else "missing",
        "release_status": "downloaded" if release_downloaded else ("missing" if release_gcs_uri else ""),
        "video_status": "downloaded" if video_downloaded else ("missing" if video_gcs_uri else ""),
        "keypoints_error": keypoints_error,
        "release_error": release_error,
        "video_error": video_error,
    }
