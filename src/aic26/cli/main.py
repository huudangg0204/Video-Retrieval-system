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
    and index into Elasticsearch.
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

@pipeline_group.command(name="init-es-index")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Delete existing index and recreate from scratch (WARNING: destructive)",
)
def init_es_index(force: bool) -> None:
    """Create or reset the Elasticsearch keyframes index."""
    from aic26.core.elasticsearch_client import ElasticsearchClient

    es = ElasticsearchClient()
    if not es.ping():
        click.echo(f"\u274c Cannot reach Elasticsearch at {settings.es_scheme}://{settings.es_host}:{settings.es_port}")
        click.echo("   Make sure Elasticsearch is running (docker compose up elasticsearch).")
        return

    if force:
        click.confirm(
            f"\u26a0\ufe0f  This will DELETE index '{settings.es_index_keyframes}' and all data. Continue?",
            abort=True,
        )

    es.ensure_keyframes_index(force_recreate=force)
    stats = es.get_stats()
    click.echo(f"\n\u2705 Elasticsearch index ready!")
    click.echo(f"   Index:      {stats['index_keyframes']}")
    click.echo(f"   Documents:  {stats['total_keyframes']}")
    click.echo(f"   Host:       {settings.es_scheme}://{settings.es_host}:{settings.es_port}")


@pipeline_group.command(name="index-to-es")
@click.option(
    "--embedding-dir", "-e",
    type=click.Path(exists=True),
    default=None,
    help="Directory with .npy EVA-02 embedding files (default: settings.embedding_dir)",
)
@click.option(
    "--keyframes-dir", "-k",
    type=click.Path(exists=True),
    default=None,
    help="Directory with keyframe images (default: settings.keyframes_dir)",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=200,
    help="Documents per bulk request",
)
@click.option(
    "--force-recreate", "-f",
    is_flag=True,
    help="Delete and recreate the ES index before indexing",
)
@click.option(
    "--no-blip",
    is_flag=True,
    help="Skip BLIP-ITM dual embedding (faster, single embedding mode)",
)
def index_to_es(
    embedding_dir: str | None,
    keyframes_dir: str | None,
    batch_size: int,
    force_recreate: bool,
    no_blip: bool,
) -> None:
    """Bulk-index pre-computed embeddings into Elasticsearch."""
    from aic26.core.elasticsearch_client import ElasticsearchClient
    from aic26.core.models import ModelRegistry
    from aic26.pipeline.indexer import ElasticsearchIndexer

    es = ElasticsearchClient()
    if not es.ping():
        click.echo(f"\u274c Cannot reach Elasticsearch at {settings.es_host}:{settings.es_port}")
        return

    click.echo(f"\u23f3 Loading models...")
    models = ModelRegistry()

    indexer = ElasticsearchIndexer(
        es_client=es,
        models=models,
        embedding_dir=embedding_dir,
        keyframes_dir=keyframes_dir,
    )

    click.echo(f"\u23f3 Indexing into Elasticsearch (force_recreate={force_recreate})...")
    stats = indexer.build_index(
        batch_size=batch_size,
        force_recreate=force_recreate,
        encode_blip=not no_blip,
    )

    click.echo(f"\n\u2705 Elasticsearch indexing complete!")
    click.echo(f"   Videos:   {stats.total_videos}")
    click.echo(f"   Frames:   {stats.total_frames}")
    click.echo(f"   Indexed:  {stats.indexed_success}")
    click.echo(f"   Failed:   {stats.indexed_failed}")
    click.echo(f"   BLIP enc: {stats.blip_encoded}")


@pipeline_group.command(name="build-index")
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Input directory containing .npy embedding files")
@click.option("--output", "-o", type=click.Path(), help="Output path for FAISS database")
@click.option("--index-type", type=click.Choice(["flat_ip", "flat_l2"]),
              default="flat_ip", help="Type of FAISS index")
def build_index(input: str, output: str | None, index_type: str) -> None:
    """[DEPRECATED] Build FAISS index. Use 'index-to-es' instead."""
    click.echo("\u26a0\ufe0f  build-index is DEPRECATED.")
    click.echo("   Use: aic26 pipeline index-to-es")



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
    from aic26.api.main import create_app
    
    host = host or settings.host
    port = port or settings.port
    debug = debug if debug is not None else settings.debug
    
    logger.info(f"Starting API server on {host}:{port}")
    
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


# ── Elasticsearch commands ─────────────────────────────────────────────

@cli.command(name="check-es")
def check_es() -> None:
    """Check Elasticsearch connection and index status."""
    from aic26.core.elasticsearch_client import ElasticsearchClient

    click.echo("=" * 60)
    click.echo("Elasticsearch Status")
    click.echo("=" * 60)

    es = ElasticsearchClient()
    url = f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}"

    if not es.ping():
        click.echo(f"\n\u274c Cannot connect to Elasticsearch at {url}")
        click.echo("   Make sure Elasticsearch is running:")
        click.echo("   docker compose -f docker/docker-compose.yml up elasticsearch -d")
        return

    click.echo(f"\n\u2705 Connected to {url}")

    info = es.get_info()
    click.echo(f"   Cluster: {info.get('cluster_name', 'N/A')}")
    click.echo(f"   Version: {info.get('version', {}).get('number', 'N/A')}")

    stats = es.get_stats()
    click.echo(f"\n[INDEX] {stats['index_keyframes']}")
    click.echo(f"   Total keyframes: {stats['total_keyframes']}")
    click.echo(f"   Total videos:    {stats['total_videos']}")
    click.echo(f"\n[SETTINGS]")
    click.echo(f"   use_dual_embedding: {settings.use_dual_embedding}")
    click.echo(f"   eva02_weight: {settings.eva02_search_weight}")
    click.echo(f"   blip_weight:  {settings.blip2_search_weight}")



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
    from pathlib import Path
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
    for pkg in ["torch", "flask", "elasticsearch", "numpy", "pandas"]:
        try:
            ver = version(pkg)
            click.echo(f"  ✅ {pkg}: {ver}")
        except Exception:
            click.echo(f"  ❌ {pkg}: NOT INSTALLED")

    click.echo(f"\nElasticsearch:")
    try:
        from aic26.core.elasticsearch_client import ElasticsearchClient
        es = ElasticsearchClient()
        if es.ping():
            es_stats = es.get_stats()
            click.echo(f"  ✅ Connected to {settings.es_host}:{settings.es_port}")
            click.echo(f"  ✅ Index '{settings.es_index_keyframes}': {es_stats['total_keyframes']} documents")
        else:
            click.echo(f"  ⚠️  Not reachable at {settings.es_host}:{settings.es_port}")
            click.echo(f"     Run: docker compose -f docker/docker-compose.yml up elasticsearch -d")
    except Exception as e:
        click.echo(f"  ❌ ES check failed: {e}")

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