import argparse
import csv
import shutil
import subprocess
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcp_fetch_utils import GCSClient


def gcloud_executable() -> str:
    executable = shutil.which("gcloud.cmd") or shutil.which("gcloud")
    if not executable:
        raise FileNotFoundError("Unable to locate gcloud or gcloud.cmd in PATH.")
    return executable


def split_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    path_without_scheme = gs_uri[5:]
    bucket_name, object_name = path_without_scheme.split("/", 1)
    return bucket_name, object_name


def list_batter_keypoint_uris(prefix_gs_uri: str) -> List[Tuple[str, str]]:
    cmd = [gcloud_executable(), "storage", "ls", "-l", f"{prefix_gs_uri}/**/batter_keypoints.csv"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return []
    matches: List[Tuple[str, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("TOTAL:"):
            continue
        parts = stripped.split()
        if len(parts) >= 3 and parts[2].startswith("gs://"):
            matches.append((parts[1], parts[2]))
    return matches


def choose_latest_uri(candidates: List[Tuple[str, str]]) -> Optional[str]:
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_segment_prefix(video_gs_uri: str, segment_id: str) -> str:
    bucket_name, object_name = split_gs_uri(video_gs_uri)
    parent_prefix = "/".join(object_name.split("/")[:-1])
    return f"gs://{bucket_name}/{parent_prefix}/{segment_id}"


def download_batter_keypoints(
    manifest_csv: str,
    summary_csv: str,
    output_dir: str,
    project_id: str = "video-backend-dev",
) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_csv)
    summary_df = pd.read_csv(summary_csv)
    success_segment_ids = set(
        summary_df.loc[summary_df["status"] == "success", "segment_id"].astype(str).tolist()
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    keypoints_dir = output_root / "batter_keypoints"
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    gcs_client = GCSClient(project_id=project_id)
    rows = []

    for _, row in manifest_df.iterrows():
        segment_id = str(row.get("segment_id", "")).strip()
        if segment_id not in success_segment_ids:
            continue

        video_id = str(row.get("Video ID", row.get("video_id", ""))).strip()
        ball = str(row.get("Ball", row.get("ball", ""))).strip()
        video_gs_uri = str(row.get("File Path", row.get("file_path", ""))).strip()
        if not video_gs_uri:
            continue

        prefix_gs_uri = build_segment_prefix(video_gs_uri, segment_id)
        candidates = list_batter_keypoint_uris(prefix_gs_uri)
        selected_gs_uri = choose_latest_uri(candidates)

        local_path = keypoints_dir / f"{segment_id}_batter_keypoints.csv"
        status = "missing"
        message = "No batter_keypoints.csv found"
        if selected_gs_uri:
            gcs_client.download_file(selected_gs_uri, str(local_path))
            status = "downloaded"
            message = f"Downloaded from {selected_gs_uri}"

        rows.append(
            {
                "video_id": video_id,
                "ball": ball,
                "segment_id": segment_id,
                "prefix_gs_uri": prefix_gs_uri,
                "batter_keypoints_gcs_uri": selected_gs_uri or "",
                "batter_keypoints_path": str(local_path) if selected_gs_uri else "",
                "status": status,
                "message": message,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_root / "batter_keypoints_download_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return result_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download batter keypoints for successful pose job runs.")
    parser.add_argument("--manifest-csv", required=True, help="Manifest CSV with Video ID, segment_id, and File Path.")
    parser.add_argument("--summary-csv", required=True, help="Pose job summary CSV with success rows.")
    parser.add_argument("--output-dir", default="trigger_detection_artifacts", help="Output directory.")
    parser.add_argument("--project-id", default="video-backend-dev", help="Google Cloud project id.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result_df = download_batter_keypoints(
        manifest_csv=args.manifest_csv,
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        project_id=args.project_id,
    )
    print(result_df[["segment_id", "status", "batter_keypoints_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
