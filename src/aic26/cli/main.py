"""
Unified CLI entry point for AIC-26.

Provides commands for:
- Pipeline: Data processing steps
- Search: Text/image/temporal search
- Serve: API server and GUI

Usage:
    aic26 --help
    aic26 pipeline run-all --video-dir ./videos
    aic26 search text --query "a red car"
    aic26 serve api --port 8080
"""
from __future__ import annotations

import click
from pathlib import Path

from aic26.config import settings
from aic26.utils.logging import setup_logging

from loguru import logger

@click.group()
@click.version_option(version="1.0.0", prog_name="AIC-26")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose (DEBUG) output",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str | None) -> None:
    """
    AIC-26: AI Challenge Video Retrieval System.
    
    Intelligent video search using EVA-02 embeddings, FAISS vector search,
    and TRAKE temporal reasoning.
    """
    ctx.ensure_object(dict)
    
    # Configure logging
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(level=log_level, log_file=settings.log_file)
    
    # Store config in context
    ctx.obj["verbose"] = verbose
    
    from loguru import logger
    logger.info(f"AIC-26 v1.0.0 starting (log level: {log_level})")


# ── Pipeline commands ──────────────────────────────────────────────────

@cli.group(name="pipeline")
def pipeline_group():
    """
    Data processing pipeline commands.
    
    Run steps of the data preparation pipeline:
    extract keyframes, detect objects, OCR, extract embeddings,
    and build the FAISS index.
    """
    pass


@pipeline_group.command(name="extract-keyframes")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing video files",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for keyframes",
)
@click.option(
    "--workers", "-w",
    type=int,
    default=None,
    help="Number of worker threads",
)
def extract_keyframes(input: str, output: str | None, workers: int | None) -> None:
    """Extract keyframes from video files."""
    from aic26.pipeline.keyframe import KeyframeExtractor
    
    extractor = KeyframeExtractor(settings)
    stats = extractor.process(
        input_dir=input,
        output_dir=output,
        num_workers=workers or settings.max_workers,
    )
    
    click.echo(f"\n✅ Keyframe extraction complete!")
    click.echo(f"   Videos processed: {stats['videos_processed']}/{stats['videos_found']}")
    click.echo(f"   Keyframes total:  {stats['keyframes_total']}")


@pipeline_group.command(name="detect-objects")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing keyframe folders",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for detection JSONs",
)
@click.option(
    "--workers", "-w",
    type=int,
    default=None,
    help="Number of worker threads",
)
def detect_objects(input: str, output: str | None, workers: int | None) -> None:
    """Run object detection on keyframes."""
    from aic26.pipeline.detection import ObjectDetector
    
    detector = ObjectDetector(settings)
    stats = detector.process(
        input_dir=input,
        output_dir=output,
        num_workers=workers or settings.max_workers,
    )
    
    click.echo(f"\n✅ Object detection complete!")
    click.echo(f"   Folders processed: {stats['folders_processed']}/{stats['folders_found']}")
    click.echo(f"   Images total:      {stats['images_total']}")


@pipeline_group.command(name="extract-embeddings")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing keyframe folders",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for .npy embedding files",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=40,
    help="Batch size for GPU processing",
)
def extract_embeddings(input: str, output: str | None, batch_size: int) -> None:
    """Extract EVA-02 embeddings from keyframes."""
    from aic26.pipeline.embedding import EmbeddingExtractor
    
    extractor = EmbeddingExtractor(settings)
    stats = extractor.process_images(
        input_dir=input,
        output_dir=output,
        batch_size=batch_size,
    )
    
    click.echo(f"\n✅ Embedding extraction complete!")
    click.echo(f"   Folders processed: {stats['folders_processed']}/{stats['folders_total']}")
    click.echo(f"   Images total:      {stats['images_total']}")


@pipeline_group.command(name="extract-ocr")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing keyframe folders",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output CSV file path",
)
@click.option(
    "--annotate/--no-annotate",
    default=False,
    help="Save annotated images with bounding boxes",
)
@click.option(
    "--min-conf",
    type=float,
    default=0.35,
    help="Minimum confidence threshold",
)
def extract_ocr(
    input: str,
    output: str | None,
    annotate: bool,
    min_conf: float,
) -> None:
    """Extract text from keyframes using OCR."""
    from aic26.pipeline.ocr import OCRExtractor
    
    ocr = OCRExtractor(settings)
    df = ocr.process(
        input_dir=input,
        output_csv=output,
        annotate=annotate,
        min_conf=min_conf,
    )
    
    click.echo(f"\n✅ OCR extraction complete!")
    click.echo(f"   Images processed: {len(df)}")
    click.echo(f"   Output: {output or settings.ocr_output_path}")

@pipeline_group.command(name="extract-caption-embeddings")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing caption JSON folders",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for .npy embedding files",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=40,
    help="Batch size for GPU processing",
)
def extract_caption_embeddings(
    input: str,
    output: str | None,
    batch_size: int,
) -> None:
    """Extract EVA-02 embeddings from caption JSON files."""
    from aic26.pipeline.embedding import EmbeddingExtractor
    
    extractor = EmbeddingExtractor(settings)
    stats = extractor.process_captions(
        input_dir=input,
        output_dir=output,
        batch_size=batch_size,
    )
    
    click.echo(f"\n✅ Caption embedding extraction complete!")
    click.echo(f"   Folders processed: {stats['folders_processed']}/{stats['folders_total']}")
    click.echo(f"   Captions total:    {stats['captions_total']}")

@pipeline_group.command(name="build-index")
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing .npy embedding files",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output path for FAISS database",
)
@click.option(
    "--index-type",
    type=click.Choice(["flat_ip", "flat_l2"]),
    default="flat_ip",
    help="Type of FAISS index",
)
def build_index(input: str, output: str | None, index_type: str) -> None:
    """Build FAISS index from embeddings."""
    from aic26.pipeline.indexer import IndexBuilder
    
    builder = IndexBuilder(settings)
    stats = builder.build(
        embedding_dir=input,
        output_path=output,
        index_type=index_type,
    )
    
    click.echo(f"\n✅ FAISS index built successfully!")
    click.echo(f"   Embeddings: {stats['embeddings_loaded']}")
    click.echo(f"   Dimension:  {stats['dimension']}")
    click.echo(f"   Index type: {stats['index_type']}")
    click.echo(f"   Output:     {stats['output_path']}")


@pipeline_group.command(name="run-all")
@click.option(
    "--video-dir", "-v",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing video files",
)
@click.option("--skip-detection", is_flag=True, help="Skip object detection step")
@click.option("--skip-ocr", is_flag=True, help="Skip OCR step")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding extraction")
@click.option("--skip-index", is_flag=True, help="Skip FAISS index building")
def run_all_pipeline(
    video_dir: str,
    skip_detection: bool,
    skip_ocr: bool,
    skip_embeddings: bool,
    skip_index: bool,
) -> None:
    """Run the complete data processing pipeline."""
    from aic26.pipeline.orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator(settings)
    results = orchestrator.run(
        video_dir=video_dir,
        skip_detection=skip_detection,
        skip_ocr=skip_ocr,
        skip_embeddings=skip_embeddings,
        skip_index=skip_index,
    )
    
    stats = results.get('_pipeline_stats', {})
    
    click.echo(f"\n{'='*60}")
    click.echo("Pipeline Complete!")
    click.echo(f"  Completed: {len(stats.get('steps_completed', []))} steps")
    click.echo(f"  Failed:    {len(stats.get('steps_failed', []))} steps")
    click.echo(f"  Skipped:   {len(stats.get('steps_skipped', []))} steps")
    click.echo(f"  Duration:  {stats.get('total_duration_seconds', 0):.1f}s")
    click.echo(f"{'='*60}")


# ── Search commands ────────────────────────────────────────────────────

@cli.group(name="search")
def search_group():
    """Search commands for video retrieval."""
    pass


@search_group.command(name="text")
@click.option("--query", "-q", required=True, help="Text query to search for")
@click.option("--top-k", "-k", type=int, default=None, help="Number of results")
@click.option("--objects", "-o", default="", help="Filter by objects")
@click.option("--threshold", "-t", type=float, default=None, help="Confidence threshold")
def search_text(query: str, top_k: int | None, objects: str, threshold: float | None) -> None:
    """Search keyframes by text description."""
    from aic26.core.retrieval import EVA02ImageRetrieval
    
    retriever = EVA02ImageRetrieval()
    results = retriever.search_text(
        query=query,
        top_k=top_k or settings.default_top_k,
        objects=objects,
        threshold=threshold or settings.default_obj_threshold,
    )
    
    if not results:
        click.echo("No results found.")
        return
    
    click.echo(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        time_str = r.get("time_str", "N/A")
        click.echo(
            f"{i:3d}. {r['video_name']:15s} | "
            f"Frame {r['frame_idx']:>6s} | "
            f"Time {time_str:>6s} | "
            f"Score: {r['similarity']:.4f}"
        )


@search_group.command(name="temporal")
@click.option("--events", "-e", multiple=True, required=True, help="Event descriptions in order")
@click.option("--mode", "-m", type=click.Choice(["closest", "highest"]), default="highest")
def search_temporal(events: tuple[str, ...], mode: str) -> None:
    """TRAKE temporal search for event sequences."""
    click.echo("⚠️  Search CLI coming in Phase 2. Use the API server for now.")


# ── Server commands ────────────────────────────────────────────────────

@cli.group(name="serve")
def serve_group():
    """Server and GUI commands."""
    pass


@serve_group.command(name="api")
@click.option("--host", "-h", default=None, help="Host address")
@click.option("--port", "-p", type=int, default=None, help="Port number")
@click.option("--debug/--no-debug", default=None, help="Enable debug mode")
def serve_api(host: str | None, port: int | None, debug: bool | None) -> None:
    """Start the REST API server."""
    from loguru import logger
    
    host = host or settings.host
    port = port or settings.port
    debug = debug if debug is not None else settings.debug
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.warning("API server will be fully migrated in Phase 4")
    logger.info("For now, use the original sever.py script")
    
    click.echo(f"⚠️  API server coming in Phase 4.")
    click.echo(f"   Would start on http://{host}:{port}")
    click.echo(f"   For now: python sever.py")


@serve_group.command(name="api")
@click.option("--host", "-h", default=None, help="Host address")
@click.option("--port", "-p", type=int, default=None, help="Port number")
@click.option("--debug/--no-debug", default=None, help="Enable debug mode")
def serve_api(host: str | None, port: int | None, debug: bool | None) -> None:
    """Start the REST API server."""
    from aic26.api.server import create_app
    
    host = host or settings.host
    port = port or settings.port
    debug = debug if debug is not None else settings.debug
    
    logger.info(f"Starting API server on http://{host}:{port}")
    
    app = create_app(settings)
    app.run(host=host, port=port, debug=debug)


@serve_group.command(name="gui")
def serve_gui() -> None:
    """Launch the desktop GUI application."""
    from aic26.gui.retrieval_app import EVA02GUI
    
    logger.info("Starting desktop GUI...")
    app = EVA02GUI(settings)
    app.run()


@serve_group.command(name="filter")
@click.option(
    "--objects-dir", "-o",
    type=click.Path(exists=True),
    help="Directory containing detection JSONs",
)
@click.option(
    "--keyframes-dir", "-k",
    type=click.Path(exists=True),
    help="Directory containing keyframe images",
)
def serve_filter(objects_dir: str | None, keyframes_dir: str | None) -> None:
    """Launch the object filter GUI."""
    import tkinter as tk
    from aic26.gui.filter_app import ObjectFilterApp
    
    objects_dir = objects_dir or str(settings.detection_dir)
    keyframes_dir = keyframes_dir or str(settings.keyframes_dir)
    
    logger.info("Starting object filter GUI...")
    root = tk.Tk()
    app = ObjectFilterApp(root, objects_dir, keyframes_dir)
    root.mainloop()


# ── Utility commands ───────────────────────────────────────────────────

@cli.command(name="config")
@click.option("--show", is_flag=True, help="Show all current settings")
def show_config(show: bool) -> None:
    """Show or validate configuration."""
    if show:
        settings.display()
    else:
        click.echo(f"Config file: .env")
        click.echo(f"Data directory: {settings.data_dir}")
        click.echo(f"GPU available: {settings.is_gpu_available()}")


@cli.command(name="check")
def check_system() -> None:
    """Check system requirements and dependencies."""
    import sys
    from importlib.metadata import version
    
    click.echo("=" * 60)
    click.echo("AIC-26 System Check")
    click.echo("=" * 60)
    
    click.echo(f"\nPython: {sys.version}")
    click.echo(f"Platform: {sys.platform}")
    
    click.echo(f"\nGPU Available: {settings.is_gpu_available()}")
    if settings.is_gpu_available():
        import torch
        click.echo(f"CUDA Version: {torch.version.cuda}")
        click.echo(f"GPU Count: {torch.cuda.device_count()}")
    
    click.echo(f"\nData Directories:")
    for dir_name in ["data_dir", "embedding_dir", "keyframes_dir"]:
        dir_path = getattr(settings, dir_name)
        exists = Path(dir_path).exists()
        icon = "✅" if exists else "⚠️"
        click.echo(f"  {icon} {dir_name}: {dir_path}")
    
    click.echo(f"\nDependencies:")
    for pkg in ["torch", "flask", "numpy", "pandas"]:
        try:
            ver = version(pkg)
            click.echo(f"  ✅ {pkg}: {ver}")
        except Exception:
            click.echo(f"  ❌ {pkg}: NOT INSTALLED")
    
    # Check faiss (can be faiss-gpu or faiss-cpu)
    try:
        ver = version("faiss-gpu")
        click.echo(f"  ✅ faiss (gpu): {ver}")
    except Exception:
        try:
            ver = version("faiss-cpu")
            click.echo(f"  ✅ faiss (cpu): {ver}")
        except Exception:
            click.echo(f"  ❌ faiss: NOT INSTALLED")

@cli.command(name="download")
@click.option(
    "--urls", "-u",
    default="urls.txt",
    type=click.Path(exists=True),
    help="File containing video URLs (one per line)",
)
@click.option(
    "--output", "-o",
    default="./videos",
    type=click.Path(),
    help="Output directory for downloaded videos",
)
@click.option(
    "--max", "-m",
    type=int,
    default=None,
    help="Maximum number of videos to download",
)
def download_videos(urls: str, output: str, max: int | None) -> None:
    """Download videos from URLs listed in a file."""
    import subprocess
    from pathlib import Path
    
    urls_path = Path(urls)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read URLs
    with open(urls_path, "r") as f:
        urls_list = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
    
    if max:
        urls_list = urls_list[:max]
    
    click.echo(f"Downloading {len(urls_list)} video(s) to {output}...")
    
    for i, url in enumerate(urls_list, 1):
        click.echo(f"[{i}/{len(urls_list)}] {url}")
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "--format", "best[height<=1080]",
                    "--output", str(output_path / "%(title)s.%(ext)s"),
                    "--no-playlist",
                    "--quiet",
                    url,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            click.echo(f"  ⚠️ Failed to download: {url}")
        except FileNotFoundError:
            click.echo("  ❌ yt-dlp not found. Install: pip install yt-dlp")
            return
    
    click.echo(f"\n✅ Download complete. Videos saved to {output}")
# ── Main entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    cli()