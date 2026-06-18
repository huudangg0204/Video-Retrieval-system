"""Script to inject ingest CLI commands into cli/main.py"""
import sys

INGEST_CODE = r'''
# ── Ingest command group ───────────────────────────────────────────────────────

@cli.group(name="ingest")
def ingest_group() -> None:
    """
    Ingest data (keyframes or videos) into Elasticsearch.

    \b
      keyframes  Pre-extracted folder/ZIP (e.g. Kaggle dataset)
      videos     Raw video files (keyframes auto-extracted first)
      auto       Auto-detect input type
    """
    pass


def _common_ingest_options(fn):
    """Attach shared options to each ingest subcommand."""
    fn = click.option("--skip-detection", is_flag=True,
                      help="Skip object detection")(fn)
    fn = click.option("--skip-ocr", is_flag=True,
                      help="Skip OCR text extraction")(fn)
    fn = click.option("--skip-es", is_flag=True,
                      help="Skip Elasticsearch indexing")(fn)
    fn = click.option("--force-recreate", "-f", is_flag=True,
                      help="Delete and recreate ES index before indexing")(fn)
    fn = click.option("--no-blip", is_flag=True,
                      help="Skip BLIP dual embedding (EVA-02 only)")(fn)
    fn = click.option("--batch-size", "-b", type=int, default=40,
                      help="GPU batch size for embedding (default: 40)")(fn)
    fn = click.option("--keyframes-dir", "-k", type=click.Path(), default=None,
                      help="Override keyframes output directory")(fn)
    return fn


def _print_ingest_stats(stats) -> None:
    """Print formatted ingestion summary table."""
    parts = ", ".join(stats.steps_completed) if stats.steps_completed else "-"
    click.echo("")
    click.echo("=" * 55)
    click.echo("Ingestion Summary")
    click.echo("=" * 55)
    click.echo(f"  Mode:            {stats.input_mode}")
    click.echo(f"  Input:           {stats.input_path}")
    click.echo(f"  Keyframe dir:    {stats.keyframe_dir}")
    click.echo(f"  Videos:          {stats.videos_found}")
    click.echo(f"  Keyframes:       {stats.keyframes_found}")
    click.echo(f"  Embeddings:      {stats.embeddings_extracted}")
    click.echo(f"  ES indexed:      {stats.es_indexed}")
    click.echo(f"  Duration:        {stats.duration_seconds:.1f}s")
    click.echo(f"  Completed:       {parts}")
    if stats.steps_failed:
        failed = ", ".join(stats.steps_failed)
        click.echo(f"  Failed:          {failed}")
    if stats.steps_skipped:
        skipped = ", ".join(stats.steps_skipped)
        click.echo(f"  Skipped:         {skipped}")
    click.echo("=" * 55)


@ingest_group.command(name="keyframes")
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Keyframe dir or .zip. Structure: input/<video_name>/<frame>.jpg")
@_common_ingest_options
def ingest_keyframes(
    input: str,
    keyframes_dir,
    skip_detection: bool,
    skip_ocr: bool,
    skip_es: bool,
    force_recreate: bool,
    no_blip: bool,
    batch_size: int,
) -> None:
    """Ingest pre-extracted keyframes (Kaggle dataset or ZIP file).

    \b
    Expected structure:
        input/
        ├── L01_V001/  <- video name
        │   ├── 000.jpg
        │   └── 001.jpg
        └── L01_V002/
    """
    from aic26.pipeline.ingest import IngestPipeline
    click.echo(f"[Ingest] keyframes | {input}")
    pipeline = IngestPipeline(
        keyframes_dir=keyframes_dir,
        batch_size=batch_size,
        skip_detection=skip_detection,
        skip_ocr=skip_ocr,
        skip_es_index=skip_es,
        force_es_recreate=force_recreate,
        encode_blip=not no_blip,
    )
    stats = pipeline.run_from_keyframes(
        input_path=input,
        output_keyframes_dir=keyframes_dir,
    )
    _print_ingest_stats(stats)


@ingest_group.command(name="videos")
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Directory of video files (.mp4, .avi, .mkv, .mov)")
@_common_ingest_options
def ingest_videos(
    input: str,
    keyframes_dir,
    skip_detection: bool,
    skip_ocr: bool,
    skip_es: bool,
    force_recreate: bool,
    no_blip: bool,
    batch_size: int,
) -> None:
    """Ingest raw video files (keyframes extracted automatically).

    \b
    Example:
        aic26 ingest videos --input data/videos/
        aic26 ingest videos --input data/videos/ --skip-detection --skip-ocr
    """
    from aic26.pipeline.ingest import IngestPipeline
    click.echo(f"[Ingest] videos | {input}")
    pipeline = IngestPipeline(
        keyframes_dir=keyframes_dir,
        batch_size=batch_size,
        skip_detection=skip_detection,
        skip_ocr=skip_ocr,
        skip_es_index=skip_es,
        force_es_recreate=force_recreate,
        encode_blip=not no_blip,
    )
    stats = pipeline.run_from_videos(video_dir=input)
    _print_ingest_stats(stats)


@ingest_group.command(name="auto")
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Input path -- auto-detects keyframes vs videos")
@_common_ingest_options
def ingest_auto(
    input: str,
    keyframes_dir,
    skip_detection: bool,
    skip_ocr: bool,
    skip_es: bool,
    force_recreate: bool,
    no_blip: bool,
    batch_size: int,
) -> None:
    """Auto-detect input type (ZIP / keyframes / videos) and ingest.

    \b
    Detection rules:
      .zip file             -> Mode A (keyframes)
      Contains .mp4/.avi    -> Mode B (videos)
      Subfolders with .jpg  -> Mode A (keyframes)
    """
    import os
    from pathlib import Path as _P
    from aic26.pipeline.ingest import IngestPipeline
    pipeline = IngestPipeline(
        keyframes_dir=keyframes_dir,
        batch_size=batch_size,
        skip_detection=skip_detection,
        skip_ocr=skip_ocr,
        skip_es_index=skip_es,
        force_es_recreate=force_recreate,
        encode_blip=not no_blip,
    )
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    if input.lower().endswith(".zip"):
        click.echo("[Auto] ZIP keyframe archive detected")
        stats = pipeline.run_from_keyframes(input_path=input)
    elif any(_P(os.path.join(input, e)).suffix.lower() in video_exts
             for e in os.listdir(input)):
        click.echo("[Auto] Video directory detected")
        stats = pipeline.run_from_videos(video_dir=input)
    else:
        click.echo("[Auto] Keyframe directory detected")
        stats = pipeline.run_from_keyframes(input_path=input)
    _print_ingest_stats(stats)

'''

TARGET = '    click.echo("   Use: aic26 pipeline index-to-es")\n\n\n'

with open("src/aic26/cli/main.py", "r", encoding="utf-8") as f:
    content = f.read()

if TARGET not in content:
    print("ERROR: marker not found in file", file=sys.stderr)
    sys.exit(1)

content = content.replace(TARGET, TARGET + INGEST_CODE, 1)

with open("src/aic26/cli/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"OK — injected {len(INGEST_CODE)} chars. Total file size: {len(content)}")
