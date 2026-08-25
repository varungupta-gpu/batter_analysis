import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


DEFAULT_API_URL = "https://video-insights-api-dev-wmoq36zjfq-el.a.run.app"
DEFAULT_CLOUD_RUN_JOB_URL = (
    "https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/"
    "namespaces/video-backend-dev/jobs/pose-overlay:run"
)

_user_token_cache: Dict[str, str] = {}


def get_user_token(api_url: str, admin_token: str, phone_number: str) -> str:
    url = f"{api_url.rstrip('/')}/api/v1/auth/admin/issue-token"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Content-Type": "application/json",
        },
        json={"phone_number": phone_number},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_or_fetch_user_token(api_url: str, admin_token: str, phone_number: str) -> str:
    if phone_number not in _user_token_cache:
        _user_token_cache[phone_number] = get_user_token(api_url, admin_token, phone_number)
        print(f"Fetched user token for {phone_number}")
    else:
        print(f"Using cached user token for {phone_number}")
    return _user_token_cache[phone_number]


def get_video_admin(api_url: str, admin_token: str, video_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/api/v1/videos/admin/{video_id}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_gcloud_access_token() -> str:
    gcloud_candidates = [
        shutil.which("gcloud"),
        shutil.which("gcloud.cmd"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
    ]
    gcloud_executable = next((candidate for candidate in gcloud_candidates if candidate and Path(candidate).exists()), None)
    if not gcloud_executable:
        raise FileNotFoundError(
            "Google Cloud SDK not found. Make sure `gcloud` is installed and available in PATH."
        )

    result = subprocess.run(
        [gcloud_executable, "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    access_token = result.stdout.strip()
    if not access_token:
        raise RuntimeError("gcloud auth print-access-token returned an empty token.")
    return access_token


def run_pose_job(
    video_id: str,
    segment_id: str,
    user_token: str,
    api_url: str,
    cloud_run_job_url: str,
    overlay_mode: str,
) -> requests.Response:
    request_id = str(uuid.uuid4())
    print(f"request_id: {request_id}")
    print(f"video_id: {video_id}")
    print(f"segment_id: {segment_id}")

    gcloud_access_token = get_gcloud_access_token()
    headers = {
        "Authorization": f"Bearer {gcloud_access_token}",
        "Content-Type": "application/json",
    }

    job_payload = {
        "request_id": request_id,
        "video_id": video_id,
        "api_base_url": api_url,
        "segment_id": segment_id,
        "bearer_access_token": user_token,
        "enable_pose_overlay_callback": False,
        "generate_overlay": overlay_mode,
    }
    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    "args": [json.dumps(job_payload)],
                    "env": [{"name": "CLEANUP_AFTER_REQUEST", "value": "false"}],
                }
            ]
        }
    }

    return requests.post(
        cloud_run_job_url,
        headers=headers,
        json=payload,
        timeout=120,
    )


def load_videos_from_json(path: str) -> List[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "videos" not in payload or not isinstance(payload["videos"], list):
        raise ValueError("JSON must contain a 'videos' array")
    return payload["videos"]


def load_videos_from_manifest(path: str, limit: int = None) -> List[dict]:
    manifest_df = pd.read_csv(path)
    if limit is not None:
        manifest_df = manifest_df.head(limit)

    videos: List[dict] = []
    for _, row in manifest_df.iterrows():
        video_id = row.get("Video ID", row.get("video_id", ""))
        segment_id = row.get("segment_id", row.get("segmentId", ""))
        ball = row.get("Ball", row.get("ball", ""))
        if pd.isna(video_id) or pd.isna(segment_id):
            continue
        videos.append(
            {
                "video_id": str(video_id).strip(),
                "segment_id": str(segment_id).strip(),
                "ball": "" if pd.isna(ball) else str(ball).strip(),
            }
        )
    return videos


def fetch_batsman_keypoints_job_style(
    admin_token: str,
    api_url: str,
    cloud_run_job_url: str,
    videos: List[dict],
    overlay_mode: str,
) -> List[dict]:
    succeeded = []
    failed = []
    results = []

    for index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"]).strip()
        segment_id = str(video["segment_id"]).strip()
        ball = str(video.get("ball", "")).strip()

        print(f"--- Segment {index}/{len(videos)} ---")
        print(f"video_id: {video_id}")
        if ball:
            print(f"ball: {ball}")
        print(f"segment_id: {segment_id}")

        status = "failed"
        message = ""
        http_status = ""
        response_body = ""
        execution_name = ""
        log_uri = ""

        try:
            details = get_video_admin(api_url, admin_token, video_id)
            owner = details.get("owner_user") or {}
            phone_number = owner.get("phone_number")
            if not phone_number:
                raise ValueError(f"No owner phone_number found for video_id {video_id}")

            user_token = get_or_fetch_user_token(api_url, admin_token, phone_number)
            response = run_pose_job(
                video_id=video_id,
                segment_id=segment_id,
                user_token=user_token,
                api_url=api_url,
                cloud_run_job_url=cloud_run_job_url,
                overlay_mode=overlay_mode,
            )
            http_status = response.status_code
            response_body = response.text

            if response.ok:
                try:
                    response_json = response.json()
                    execution_name = str(response_json.get("metadata", {}).get("name", "") or "")
                    log_uri = str(response_json.get("status", {}).get("logUri", "") or "")
                except Exception:
                    execution_name = ""
                    log_uri = ""
                status = "success"
                message = "Pose job triggered successfully"
                print(f"SUCCESS: {message}")
                if execution_name:
                    print(f"execution_name: {execution_name}")
                if log_uri:
                    print(f"log_uri: {log_uri}")
            else:
                message = f"HTTP {response.status_code}"
                print(f"FAILED: {message}")
                print("Response Body:")
                print(response_body)

        except requests.exceptions.RequestException as exc:
            message = str(exc)
            print(f"FAILED: {message}")
        except subprocess.CalledProcessError as exc:
            message = f"gcloud exited with code {exc.returncode}"
            print(f"FAILED: {message}")
        except Exception as exc:
            message = str(exc)
            print(f"FAILED: {message}")

        row = {
            "video_id": video_id,
            "ball": ball,
            "segment_id": segment_id,
            "status": status,
            "message": message,
            "http_status": http_status,
            "response_body": response_body,
            "execution_name": execution_name,
            "log_uri": log_uri,
        }
        results.append(row)
        if status == "success":
            succeeded.append(row)
        else:
            failed.append(row)
        print("")

    print("=" * 50)
    print(f"SUMMARY: {len(succeeded)} succeeded, {len(failed)} failed out of {len(videos)} total")
    print("=" * 50)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger batsman pose extraction one segment at a time using the pose-overlay job style."
    )
    parser.add_argument("--admin-token", required=True, help="Admin access token.")
    parser.add_argument("--input-json", default="", help="JSON file with a 'videos' array of video_id and segment_id.")
    parser.add_argument("--manifest-csv", default="", help="CSV with Video ID and segment_id columns.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base API URL.")
    parser.add_argument("--cloud-run-job-url", default=DEFAULT_CLOUD_RUN_JOB_URL, help="Cloud Run job trigger URL.")
    parser.add_argument("--overlay-mode", default="both", help="Value to send as generate_overlay.")
    parser.add_argument("--output-dir", default="trigger_detection_artifacts", help="Directory for summary output.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.input_json and not args.manifest_csv:
        raise ValueError("Pass either --input-json or --manifest-csv.")

    if args.input_json:
        videos = load_videos_from_json(args.input_json)
        if args.limit is not None:
            videos = videos[: args.limit]
    else:
        videos = load_videos_from_manifest(args.manifest_csv, limit=args.limit)

    results = fetch_batsman_keypoints_job_style(
        admin_token=args.admin_token,
        api_url=args.api_url,
        cloud_run_job_url=args.cloud_run_job_url,
        videos=videos,
        overlay_mode=args.overlay_mode,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_root / "batsman_pose_job_summary.csv", index=False)


if __name__ == "__main__":
    main()
