# src/core/pipeline.py
"""Pipeline orchestrator with memory management."""

import gc
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .registry import ComponentRegistry
from .memory_manager import MemoryManager
from ..contracts.importer import IImporter
from ..contracts.normalizer import INormalizer
from ..contracts.classifier import IClassifier
from ..contracts.refiner import IRefiner
from ..contracts.vectorizer import IVectorizer
from ..contracts.storage import IStorage
from ..models.normalized import NormalizedMessage
from ..models.processed import ProcessedMessage


console = Console()


class Pipeline:
    """
    Orchestrates the full ingestion pipeline with memory-efficient processing.

    Stages:
    1. Import: Read source and yield NormalizedMessage
    2. Normalize: Clean content for UI and vectorization
    3. Classify: Determine intent and extract topics (GPU Stage 1)
    4. Refine: Extract semantic intent using SLM (GPU Stage 2) [Optional]
    5. Vectorize: Generate embeddings (GPU Stage 3)
    6. Persist: Store in database

    Memory Management:
    - Context-aware MemoryManager for GPU resource optimization
    - Avoids thrashing by keeping models loaded when needed by next stage
    - GC + CUDA cache clearing between unrelated stages
    - Memory profiling logged to memory_profile.log
    """

    def __init__(self, config: Dict[str, Any], registry: ComponentRegistry):
        self.config = config
        self.registry = registry
        self.memory_log: List[Dict] = []
        self.memory_manager = MemoryManager(config.get('memory', {}))

    def process_source(self, source_path: Path) -> Dict[str, Any]:
        """
        Process a data source through the complete pipeline.

        Args:
            source_path: Path to source file/ZIP

        Returns:
            Statistics dictionary
        """
        console.print(f"\n[bold cyan]Processing:[/bold cyan] {source_path}")

        start_time = time.time()
        stats = {
            'source': str(source_path),
            'start_time': datetime.now().isoformat(),
            'messages_imported': 0,
            'messages_stored': 0,
            'errors': [],
        }

        try:
            # Stage 1: Import
            normalized_messages = list(self._stage_import(source_path))
            stats['messages_imported'] = len(normalized_messages)

            if not normalized_messages:
                console.print("[yellow]⚠ No messages found in source[/yellow]")
                return stats

            # Stage 2: Normalize
            normalized_messages = self._stage_normalize(normalized_messages)

            # Stage 3: Classify (GPU Stage 1)
            classified_messages = self._stage_classify(normalized_messages)
            self._cleanup_memory("After Classification")

            # Stage 4: Refine (GPU Stage 2) - Optional
            refined_messages = self._stage_refine(classified_messages)
            # Note: No cleanup here if vectorizer uses same GPU (context-aware)

            # Stage 5: Vectorize (GPU Stage 3)
            vectorized_messages = self._stage_vectorize(refined_messages)
            self._cleanup_memory("After Vectorization")

            # Stage 6: Persist
            message_ids = self._stage_persist(vectorized_messages)
            stats['messages_stored'] = len(message_ids)

            # Final stats
            elapsed = time.time() - start_time
            stats['end_time'] = datetime.now().isoformat()
            stats['elapsed_seconds'] = elapsed
            stats['messages_per_second'] = stats['messages_stored'] / elapsed if elapsed > 0 else 0

            console.print(f"\n[bold green]✓ Completed[/bold green] in {elapsed:.1f}s")
            console.print(f"  Messages: {stats['messages_stored']}")
            console.print(f"  Speed: {stats['messages_per_second']:.1f} msgs/sec")

            # Write memory log
            self._write_memory_log()

        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {e}")
            stats['errors'].append(str(e))
            raise

        return stats

    def _stage_import(self, source_path: Path) -> Iterator[NormalizedMessage]:
        """Stage 1: Import data from source."""
        console.print("\n[bold]Stage 1: Import[/bold]")

        # Get importer from config
        importer_config = self.config['components']['importers'][0]  # First configured importer
        importer: IImporter = self.registry.create_instance(
            name='importer',
            class_path=importer_config['class'],
            config=importer_config.get('config', {})
        )

        # Get metadata
        try:
            metadata = importer.get_metadata(source_path)
            console.print(f"  Format: {metadata.get('format', 'unknown')}")
            console.print(f"  Conversations: {metadata.get('total_conversations', 0)}")
        except Exception as e:
            console.print(f"  [yellow]Could not read metadata: {e}[/yellow]")

        # Import messages
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Importing messages...", total=None)

            messages = []
            for message in importer.import_data(source_path):
                messages.append(message)
                progress.update(task, advance=1)

            progress.update(task, completed=len(messages))

        console.print(f"  [green]✓[/green] Imported {len(messages)} messages")
        return iter(messages)

    def _stage_normalize(self, messages: List[NormalizedMessage]) -> List[tuple]:
        """
        Stage 2: Normalize content.

        Returns: List of (NormalizedMessage, clean_content, vectorizable_content)
        """
        console.print("\n[bold]Stage 2: Normalize[/bold]")

        # Get normalizer
        normalizer_config = self.config['components']['normalizer']
        normalizer: INormalizer = self.registry.create_instance(
            name='normalizer',
            class_path=normalizer_config['class'],
            config=normalizer_config.get('config', {})
        )

        normalized = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Normalizing content...", total=len(messages))

            for msg in messages:
                # Clean for UI
                clean = normalizer.normalize(
                    msg.content,
                    content_type=msg.content_type
                )

                # Prepare for vectorization (soft sanitization)
                vectorizable = normalizer.prepare_for_vectorization(clean)

                normalized.append((msg, clean, vectorizable))
                progress.update(task, advance=1)

        console.print(f"  [green]✓[/green] Normalized {len(normalized)} messages")
        return normalized

    def _stage_classify(self, normalized: List[tuple]) -> List[tuple]:
        """
        Stage 3: Classify intent (GPU Stage 1).

        Returns: List of (NormalizedMessage, clean_content, vectorizable_content, Classification)
        """
        # Check if classifier is configured (optional stage)
        if 'classifier' not in self.config.get('components', {}):
            console.print("\n[bold]Stage 3: Classify[/bold] [dim](skipped - not configured)[/dim]")
            # Return with default classification
            from ..models.processed import Classification
            default_classification = Classification(
                intent='unknown',
                confidence=0.0,
                topics=[],
                is_question=False,
                is_command=False
            )
            return [(msg, clean, vectorizable, default_classification) for msg, clean, vectorizable in normalized]

        console.print("\n[bold]Stage 3: Classify[/bold]")
        self._log_memory("Before Classifier Load")

        # Get classifier
        classifier_config = self.config['components']['classifier']
        classifier: IClassifier = self.registry.create_instance(
            name='classifier',
            class_path=classifier_config['class'],
            config=classifier_config.get('config', {})
        )

        # Load model
        classifier.load()
        self._log_memory("After Classifier Load")

        classified = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Classifying messages...", total=len(normalized))

            # Batch classification
            batch_size = classifier_config.get('config', {}).get('batch_size', 32)

            for i in range(0, len(normalized), batch_size):
                batch = normalized[i:i + batch_size]

                # Extract texts for classification
                texts = [item[2] for item in batch]  # vectorizable_content

                # Classify batch
                classifications = classifier.classify_batch(texts)

                # Combine results
                for (msg, clean, vectorizable), classification in zip(batch, classifications):
                    classified.append((msg, clean, vectorizable, classification))

                progress.update(task, advance=len(batch))

        # Unload model
        classifier.unload()
        self._log_memory("After Classifier Unload")

        console.print(f"  [green]✓[/green] Classified {len(classified)} messages")
        return classified

    def _stage_refine(self, classified: List[tuple]) -> List[tuple]:
        """
        Stage 4: Semantic Refinement using SLM (GPU Stage 2).

        Extracts semantic intent from vectorizable_content using a local SLM.
        This produces refined_content which is semantically dense and optimized
        for vectorization.

        Returns: List of (NormalizedMessage, clean_content, vectorizable_content, Classification, refined_content)
        """
        # Check if refiner is configured
        if 'refiner' not in self.config.get('components', {}):
            console.print("\n[bold]Stage 4: Refine[/bold] [dim](skipped - not configured)[/dim]")
            # Pass through without refinement - use vectorizable_content as refined
            return [
                (msg, clean, vectorizable, classification, vectorizable)
                for msg, clean, vectorizable, classification in classified
            ]

        console.print("\n[bold]Stage 4: Refine[/bold]")
        self._log_memory("Before Refiner Load")

        # Get refiner
        refiner_config = self.config['components']['refiner']
        refiner: IRefiner = self.registry.create_instance(
            name='refiner',
            class_path=refiner_config['class'],
            config=refiner_config.get('config', {})
        )

        # Load model
        refiner.load()
        self._log_memory("After Refiner Load")

        refined = []
        batch_size = refiner_config.get('config', {}).get('batch_size', 8)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Refining messages...", total=len(classified))

            for i in range(0, len(classified), batch_size):
                batch = classified[i:i + batch_size]

                # Extract texts for refinement
                texts = [item[2] for item in batch]  # vectorizable_content

                # Refine batch
                refinements = refiner.refine_batch(texts)

                # Combine results
                for (msg, clean, vectorizable, classification), refinement in zip(batch, refinements):
                    refined.append((
                        msg, clean, vectorizable, classification,
                        refinement.refined_content
                    ))

                progress.update(task, advance=len(batch))

        # Unload model (unless next stage can share GPU)
        # For now, unload - MemoryManager integration can optimize later
        refiner.unload()
        self._log_memory("After Refiner Unload")

        console.print(f"  [green]✓[/green] Refined {len(refined)} messages")
        return refined

    def _stage_vectorize(self, refined: List[tuple]) -> List[ProcessedMessage]:
        """
        Stage 5: Generate embeddings (GPU Stage 3).

        Vectorizes the refined_content (semantic intent) instead of raw text.

        Returns: List of ProcessedMessage with vectors
        """
        console.print("\n[bold]Stage 5: Vectorize[/bold]")
        self._log_memory("Before Vectorizer Load")

        # Get vectorizer
        vectorizer_config = self.config['components']['vectorizer']
        vectorizer: IVectorizer = self.registry.create_instance(
            name='vectorizer',
            class_path=vectorizer_config['class'],
            config=vectorizer_config.get('config', {})
        )

        # Load model
        vectorizer.load()
        self._log_memory("After Vectorizer Load")

        processed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Vectorizing messages...", total=len(refined))

            # Batch vectorization
            batch_size = vectorizer_config.get('config', {}).get('batch_size', 64)

            for i in range(0, len(refined), batch_size):
                batch = refined[i:i + batch_size]

                # Extract refined_content for vectorization (index 4)
                texts = [item[4] for item in batch]  # refined_content

                # Vectorize batch
                vectors = vectorizer.vectorize_batch(texts)

                # Create ProcessedMessage objects
                for (msg, clean, vectorizable, classification, refined_content), vector in zip(batch, vectors):
                    processed_msg = ProcessedMessage(
                        normalized=msg,
                        clean_content=clean,
                        vectorizable_content=vectorizable,
                        classification=classification,
                        vector=vector,
                        refined_content=refined_content,
                    )
                    processed.append(processed_msg)

                progress.update(task, advance=len(batch))

        # Unload model
        vectorizer.unload()
        self._log_memory("After Vectorizer Unload")

        console.print(f"  [green]✓[/green] Vectorized {len(processed)} messages")
        return processed

    def _stage_persist(self, messages: List[ProcessedMessage]) -> List[str]:
        """Stage 6: Store in database."""
        console.print("\n[bold]Stage 6: Persist[/bold]")

        # Get storage
        storage_config = self.config['components']['storage']
        storage: IStorage = self.registry.create_instance(
            name='storage',
            class_path=storage_config['class'],
            config=storage_config.get('config', {})
        )

        # Initialize if needed
        storage.initialize()

        # Save batch
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Saving to database...", total=len(messages))

            message_ids = storage.save_messages_batch(messages)
            progress.update(task, completed=len(messages))

        console.print(f"  [green]✓[/green] Saved {len(message_ids)} messages")

        # Print stats
        stats = storage.get_statistics()
        console.print(f"\n[bold]Database Statistics:[/bold]")
        console.print(f"  Total messages: {stats['total_messages']}")
        console.print(f"  Total conversations: {stats['total_conversations']}")
        console.print(f"  Size: {stats['size_mb']:.1f} MB")

        return message_ids

    def _cleanup_memory(self, stage_name: str):
        """Force garbage collection and CUDA cache clearing."""
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._log_memory(stage_name)

    def _log_memory(self, stage: str):
        """Log memory usage."""
        entry = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            entry['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            entry['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        self.memory_log.append(entry)

    def _write_memory_log(self):
        """Write memory log to file."""
        log_path = Path("data/memory_profile.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w') as f:
            f.write("Memory Profile Log\n")
            f.write("=" * 80 + "\n\n")

            for entry in self.memory_log:
                f.write(f"Stage: {entry['stage']}\n")
                f.write(f"  Time: {entry['timestamp']}\n")

                if 'cuda_allocated_mb' in entry:
                    f.write(f"  CUDA Allocated: {entry['cuda_allocated_mb']:.1f} MB\n")
                    f.write(f"  CUDA Reserved: {entry['cuda_reserved_mb']:.1f} MB\n")

                f.write("\n")

        console.print(f"\n[dim]Memory profile saved to: {log_path}[/dim]")
