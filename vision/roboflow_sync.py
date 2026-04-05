"""
training/roboflow_sync.py - pong-roboflow: Roboflow integration for CV data.

Manages the lifecycle of images between the local data directory and a
Roboflow project: upload new images, pull labelled data with train/val/test
splits, and sync deletions.

Directory layout after pull:
    data/
      images/            ← unprocessed images (not yet uploaded or labelled)
      train/images/      ← labelled train images (moved from data/images/)
      train/labels/      ← YOLO .txt labels
      valid/images/      ← labelled validation images
      valid/labels/
      test/images/       ← labelled test images
      test/labels/

Configuration:
    data/.roboflow.json  - API key, workspace slug, project slug
    data/.roboflow_manifest.json - local tracking of uploaded images

Usage:
    pong-roboflow upload                    Upload new images to Roboflow
    pong-roboflow pull --version 1          Download labels + organise splits
    pong-roboflow sync                      Remove locally what was deleted in Roboflow
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SPLITS = {"train", "valid", "test"}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(
    api_key: str | None,
    workspace: str | None,
    project: str | None,
    config_path: Path | None = None,
) -> dict[str, str]:
    """
    Load Roboflow credentials from .roboflow.json, with CLI overrides.
    """
    config: dict[str, str] = {}
    if config_path and config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", config_path, e)

    result = {
        "api_key": api_key or config.get("api_key", ""),
        "workspace": workspace or config.get("workspace", ""),
        "project": project or config.get("project", ""),
    }
    missing = [k for k, v in result.items() if not v]
    if missing:
        raise click.ClickException(
            f"Missing Roboflow config: {', '.join(missing)}. "
            f"Set in {config_path or '.roboflow.json'} or pass via --api-key/--workspace/--project."
        )
    return result


def _get_project(config: dict[str, str]):
    """Return a roboflow Project object from the SDK."""
    import roboflow

    rf = roboflow.Roboflow(api_key=config["api_key"])
    ws = rf.workspace(config["workspace"])
    return ws.project(config["project"])


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load manifest: {filename: {"id": "...", "split": "train"|null}}."""
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(manifest: dict[str, dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Roboflow listing helper
# ---------------------------------------------------------------------------


def _list_remote_images(project) -> dict[str, dict]:
    """
    List all images in the Roboflow project.

    Returns {filename: {"id": "...", "split": "train"|"valid"|"test"|None}}.
    """
    remote: dict[str, dict] = {}
    try:
        for page in project.search_all(
            fields=["id", "name", "split"],
            limit=250,
        ):
            for img in page:
                name = img.get("name", "")
                remote[name] = {
                    "id": img.get("id", ""),
                    "split": img.get("split"),
                }
    except Exception as e:
        raise click.ClickException(f"Failed to list Roboflow images: {e}")
    return remote


# ---------------------------------------------------------------------------
# shots.jsonl helpers
# ---------------------------------------------------------------------------


def _update_shots_jsonl(path_remap: dict[str, str], shots_jsonl: Path) -> int:
    """
    Remap image paths in shots.jsonl. Returns count of remapped records.

    path_remap: {old_relative_path: new_relative_path}
    """
    if not shots_jsonl.exists() or not path_remap:
        return 0

    lines: list[str] = []
    remapped = 0
    with open(shots_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            old_path = r.get("image", "")
            if old_path in path_remap:
                r["image"] = path_remap[old_path]
                remapped += 1
            lines.append(json.dumps(r))

    tmp = shots_jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(str(tmp), str(shots_jsonl))
    return remapped


def _remove_from_shots_jsonl(filenames: set[str], shots_jsonl: Path) -> int:
    """
    Remove records from shots.jsonl whose image path ends with any filename
    in the given set. Returns count of removed records.
    """
    if not shots_jsonl.exists() or not filenames:
        return 0

    lines: list[str] = []
    removed = 0
    with open(shots_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img = r.get("image", "")
            basename = Path(img).name
            if basename in filenames:
                removed += 1
                continue
            lines.append(json.dumps(r))

    tmp = shots_jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(str(tmp), str(shots_jsonl))
    return removed


# ---------------------------------------------------------------------------
# Image location helpers
# ---------------------------------------------------------------------------


def _find_image(filename: str, data_dir: Path) -> Path | None:
    """
    Find an image file across data/images/ and all split directories.
    Returns the first match, or None.
    """
    candidates = [
        data_dir / "images" / filename,
    ]
    for split in VALID_SPLITS:
        candidates.append(data_dir / split / "images" / filename)

    for p in candidates:
        if p.exists():
            return p
    return None


def _image_rel_path(abs_path: Path, data_dir: Path) -> str:
    """Return the image path relative to data_dir (e.g. 'train/images/foo.jpg')."""
    try:
        return str(abs_path.relative_to(data_dir))
    except ValueError:
        return str(abs_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("pong-roboflow")
@click.option("--elastics", required=True, type=int,
              help="Number of elastics on the launcher (e.g. 2 or 4). "
                   "Determines data directory: data/{N}_elastics/")
@click.option("--api-key", default=None, envvar="ROBOFLOW_API_KEY",
              help="Roboflow API key (overrides config file).")
@click.option("--workspace", default=None, envvar="ROBOFLOW_WORKSPACE",
              help="Roboflow workspace slug (overrides config file).")
@click.option("--project", default=None, envvar="ROBOFLOW_PROJECT",
              help="Roboflow project slug (overrides config file).")
@click.pass_context
def cli(ctx: click.Context, elastics: int, api_key: str | None,
        workspace: str | None, project: str | None) -> None:
    """Manage images and labels between local data/ and Roboflow."""
    from utils.data_dir import elastics_data_dir
    data_dir = elastics_data_dir(elastics)
    data_dir.mkdir(parents=True, exist_ok=True)
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["workspace"] = workspace
    ctx.obj["project"] = project
    ctx.obj["data_dir"] = data_dir
    ctx.obj["config_path"] = data_dir / ".roboflow.json"
    ctx.obj["manifest_path"] = data_dir / ".roboflow_manifest.json"
    ctx.obj["shots_jsonl"] = data_dir / "shots.jsonl"


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@cli.command("upload")
@click.option("--batch", default=None, help="Roboflow batch name to group uploads.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be uploaded without uploading.")
@click.pass_context
def upload_cmd(ctx: click.Context, batch: str | None, dry_run: bool) -> None:
    """Upload new images from data/{N}_elastics/images/ to Roboflow."""
    from tqdm import tqdm

    data_dir = ctx.obj["data_dir"]
    config = _load_config(
        ctx.obj["api_key"], ctx.obj["workspace"], ctx.obj["project"],
        ctx.obj["config_path"],
    )
    project = _get_project(config)
    manifest = _load_manifest(ctx.obj["manifest_path"])
    image_path = data_dir / "images"

    if not image_path.exists():
        raise click.ClickException(f"Image directory not found: {image_path}")

    local_files = sorted(p.name for p in image_path.glob("*.jpg"))
    click.echo(f"Local images in {image_path}: {len(local_files)}")

    # Bootstrap: list all remote images and add to manifest
    click.echo("Fetching remote image list...")
    remote = _list_remote_images(project)
    click.echo(f"Remote images in Roboflow: {len(remote)}")

    bootstrapped = 0
    for name, info in remote.items():
        if name not in manifest:
            manifest[name] = info
            bootstrapped += 1
        else:
            # Update split info from remote
            manifest[name]["split"] = info.get("split")

    if bootstrapped:
        click.echo(f"Bootstrapped {bootstrapped} existing remote images into manifest")

    # Find images to upload: in local dir but not in manifest
    to_upload = [f for f in local_files if f not in manifest]
    already = len(local_files) - len(to_upload)

    click.echo(f"To upload: {len(to_upload)}  Already uploaded: {already}")

    if dry_run:
        for f in to_upload:
            click.echo(f"  Would upload: {f}")
        _save_manifest(manifest, ctx.obj["manifest_path"])
        return

    if not to_upload:
        click.echo("Nothing to upload.")
        _save_manifest(manifest, ctx.obj["manifest_path"])
        return

    uploaded = 0
    errors = 0
    with tqdm(to_upload, unit="img") as pbar:
        for filename in pbar:
            pbar.set_postfix(file=filename[:25])
            path = image_path / filename
            try:
                result = project.single_upload(
                    image_path=str(path),
                    batch_name=batch,
                    num_retry_uploads=3,
                )
                # Extract image ID from response
                img_id = ""
                if isinstance(result, dict):
                    img_id = result.get("id", "") or result.get("image", {}).get("id", "")

                manifest[filename] = {"id": img_id, "split": None}
                uploaded += 1
            except Exception as e:
                pbar.write(f"  ERROR uploading {filename}: {e}")
                errors += 1

    _save_manifest(manifest, ctx.obj["manifest_path"])
    click.echo(f"\nUpload complete. Uploaded: {uploaded}  Errors: {errors}")


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


@cli.command("pull")
@click.option("--version", "version_num", required=True, type=int,
              help="Roboflow version number to download.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Overwrite existing label files.")
@click.pass_context
def pull_cmd(ctx: click.Context, version_num: int, overwrite: bool) -> None:
    """Download YOLO labels from Roboflow and organise images into splits."""
    data = ctx.obj["data_dir"]
    config = _load_config(
        ctx.obj["api_key"], ctx.obj["workspace"], ctx.obj["project"],
        ctx.obj["config_path"],
    )
    project = _get_project(config)
    manifest = _load_manifest(ctx.obj["manifest_path"])

    # Refresh manifest splits from Roboflow
    click.echo("Refreshing manifest from Roboflow...")
    remote = _list_remote_images(project)
    for name, info in remote.items():
        if name in manifest:
            manifest[name]["split"] = info.get("split")
        else:
            manifest[name] = info

    # Download version export
    click.echo(f"Downloading version {version_num} (yolov5pytorch)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        import re

        version = project.version(version_num)
        # location must be a non-existing path - the SDK skips download
        # if the directory already exists.
        dl_path = str(Path(tmp_dir) / "download")
        dataset = version.download("yolov5pytorch", location=dl_path)

        # Build label and image lookups: {original_stem: label_text/image_path}
        # Roboflow renames files to {stem}_jpg.rf.{hash}.{ext} - strip the
        # suffix to recover the original filename stem.
        tmp = Path(dl_path)
        label_lookup: dict[str, str] = {}
        image_lookup: dict[str, Path] = {}
        for split_name in ("train", "valid", "test"):
            labels_dir = tmp / split_name / "labels"
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    if label_file.name == "classes.txt":
                        continue
                    # "20260327_012608_411127_jpg.rf.abc123" → "20260327_012608_411127"
                    original_stem = re.sub(
                        r"_jpg\.rf\.[a-f0-9]+$", "", label_file.stem
                    )
                    label_lookup[original_stem] = label_file.read_text()

            images_dir = tmp / split_name / "images"
            if images_dir.exists():
                for img_file in images_dir.glob("*.jpg"):
                    original_stem = re.sub(
                        r"_jpg\.rf\.[a-f0-9]+$", "", img_file.stem
                    )
                    image_lookup[original_stem] = img_file

        click.echo(f"Labels found in export: {len(label_lookup)}")
        click.echo(f"Images found in export: {len(image_lookup)}")

        # Move/copy images and write labels (inside the tempdir block so
        # export images are still available for fallback copy).
        moved = 0
        copied_from_export = 0
        split_changed = 0
        labels_written = 0
        unassigned = 0
        path_remap: dict[str, str] = {}  # old_rel → new_rel for shots.jsonl

        for filename, info in manifest.items():
            split = info.get("split")
            stem = Path(filename).stem

            if split not in VALID_SPLITS:
                unassigned += 1
                continue

            if stem not in label_lookup:
                continue

            # Target locations
            target_img_dir = data / split / "images"
            target_lbl_dir = data / split / "labels"
            target_img = target_img_dir / filename
            target_lbl = target_lbl_dir / (stem + ".txt")

            # Find source image - prefer local, fall back to export copy
            source_img = _find_image(filename, data)

            if source_img is not None and source_img != target_img:
                # Image exists locally but not in the right place - move it
                target_img_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_img), str(target_img))

                old_rel = _image_rel_path(source_img, data)
                new_rel = _image_rel_path(target_img, data)
                path_remap[old_rel] = new_rel

                if source_img.parent.parent.name in VALID_SPLITS:
                    split_changed += 1
                    # Clean up old label if split changed
                    old_lbl = source_img.parent.parent / "labels" / (stem + ".txt")
                    if old_lbl.exists():
                        old_lbl.unlink()
                else:
                    moved += 1
            elif source_img is None and stem in image_lookup:
                # Image not found locally - copy from export
                target_img_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(image_lookup[stem]), str(target_img))
                copied_from_export += 1
            elif source_img is None:
                logger.debug("Image not found locally or in export: %s", filename)
                continue

            # Write label
            if not target_lbl.exists() or overwrite:
                target_lbl_dir.mkdir(parents=True, exist_ok=True)
                target_lbl.write_text(label_lookup[stem])
                labels_written += 1

    # Update shots.jsonl paths
    remapped = (
        _update_shots_jsonl(path_remap, ctx.obj["shots_jsonl"])
        if path_remap
        else 0
    )

    _save_manifest(manifest, ctx.obj["manifest_path"])

    click.echo(f"\nPull complete:")
    click.echo(f"  Images moved to splits:   {moved}")
    click.echo(f"  Images copied from export: {copied_from_export}")
    click.echo(f"  Split changes:             {split_changed}")
    click.echo(f"  Labels written:            {labels_written}")
    click.echo(f"  shots.jsonl remapped:      {remapped}")
    click.echo(f"  Unassigned (no split):     {unassigned}")


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@cli.command("sync")
@click.option("--auto-delete", is_flag=True, default=False,
              help="Delete without prompting.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be deleted without deleting.")
@click.pass_context
def sync_cmd(ctx: click.Context, auto_delete: bool, dry_run: bool) -> None:
    """Detect images deleted from Roboflow and remove them locally."""
    data = ctx.obj["data_dir"]
    config = _load_config(
        ctx.obj["api_key"], ctx.obj["workspace"], ctx.obj["project"],
        ctx.obj["config_path"],
    )
    project = _get_project(config)
    manifest = _load_manifest(ctx.obj["manifest_path"])

    # Fetch current remote state
    click.echo("Fetching remote image list...")
    remote = _list_remote_images(project)
    click.echo(f"Remote images: {len(remote)}  Manifest entries: {len(manifest)}")

    # Update splits for images that still exist
    for name in list(manifest):
        if name in remote:
            manifest[name]["split"] = remote[name].get("split")

    # Find deletions: in manifest but not in remote
    deleted = {f for f in manifest if f not in remote}

    if not deleted:
        click.echo("No deletions detected.")
        _save_manifest(manifest, ctx.obj["manifest_path"])
        return

    click.echo(f"\nFound {len(deleted)} image(s) deleted from Roboflow:")

    images_deleted = 0
    labels_deleted = 0
    filenames_to_remove_from_shots: set[str] = set()

    for filename in sorted(deleted):
        img_path = _find_image(filename, data)
        location = str(img_path) if img_path else "(not found locally)"

        if dry_run:
            click.echo(f"  Would delete: {filename}  ({location})")
            continue

        if not auto_delete:
            response = click.prompt(
                f"  Delete {filename} ({location})? [y/N]",
                default="n",
                show_default=False,
            )
            if response.lower() != "y":
                continue

        # Delete image
        if img_path and img_path.exists():
            img_path.unlink()
            images_deleted += 1
            click.echo(f"  Deleted image: {img_path}")

            # Delete corresponding label
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                label_path.unlink()
                labels_deleted += 1
                click.echo(f"  Deleted label: {label_path}")

        # Also check flat data/labels/ for legacy labels
        legacy_label = data / "labels" / (Path(filename).stem + ".txt")
        if legacy_label.exists():
            legacy_label.unlink()
            labels_deleted += 1

        filenames_to_remove_from_shots.add(filename)
        del manifest[filename]

    # Remove from shots.jsonl
    shots_removed = 0
    if not dry_run and filenames_to_remove_from_shots:
        shots_removed = _remove_from_shots_jsonl(
            filenames_to_remove_from_shots, ctx.obj["shots_jsonl"]
        )

    _save_manifest(manifest, ctx.obj["manifest_path"])

    if dry_run:
        click.echo(f"\nDry run - no changes made. Would delete {len(deleted)} images.")
    else:
        click.echo(f"\nSync complete:")
        click.echo(f"  Images deleted:           {images_deleted}")
        click.echo(f"  Labels deleted:           {labels_deleted}")
        click.echo(f"  shots.jsonl records removed: {shots_removed}")


if __name__ == "__main__":
    cli()
