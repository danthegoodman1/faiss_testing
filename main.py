"""Distributed FAISS demo with shard construction and lossless reranking."""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np


DIMS = 128
SHARDS = 3
VECTORS_PER_SHARD = 100_000
TEST_VECTORS_COUNT = 10
TOP_K = 5
CANDIDATES_PER_SHARD = 20  # Increased from 20 to improve recall
NLIST = 1000  # Number of clusters for IVF (restored to 1000 for better granularity)
NPROBE = NLIST  # Probe all clusters to maximize recall (tune down for speed experiments)

DATA_ROOT = Path(__file__).resolve().parent


def shard_id_range(shard_id: int) -> tuple[int, int]:
    """Return (start_id, end_i`d) for a shard's global ID range (end_id exclusive)."""
    start = shard_id * VECTORS_PER_SHARD
    end = start + VECTORS_PER_SHARD
    return start, end


def global_to_local(global_id: int, shard_id: int) -> int:
    """Convert global ID to local shard ID."""
    start, _ = shard_id_range(shard_id)
    return global_id - start


def local_to_global(local_id: int, shard_id: int) -> int:
    """Convert local shard ID to global ID."""
    start, _ = shard_id_range(shard_id)
    return start + local_id


def shard_index_path(shard_id: int) -> Path:
    return DATA_ROOT / f"shard_{shard_id}.faiss"


def test_vectors_path() -> Path:
    return DATA_ROOT / "test_vectors.npy"


def generate_random_vectors(count: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.random((count, dims), dtype=np.float32)
    return vectors.astype(np.float32, copy=False)


def build_shards() -> None:
    print(
        f"Building {SHARDS} shards with {VECTORS_PER_SHARD} vectors each NLIST={NLIST}..."
    )

    total_start = time.perf_counter()

    for shard_id in range(SHARDS):
        shard_start = time.perf_counter()

        # Use IndexIVFFlat - clustering without lossy quantization
        quantizer = faiss.IndexFlatIP(DIMS)
        index = faiss.IndexIVFFlat(quantizer, DIMS, NLIST, faiss.METRIC_INNER_PRODUCT)
        shard_vectors = generate_random_vectors(
            VECTORS_PER_SHARD, DIMS, shard_id * 1000
        )

        # Train the index first, then add vectors
        index.train(shard_vectors)
        index.make_direct_map()  # Enable direct map for reconstruct() support
        index.add(shard_vectors)

        index_path = shard_index_path(shard_id)

        if index_path.exists():
            index_path.unlink()

        faiss.write_index(index, str(index_path))

        shard_elapsed = time.perf_counter() - shard_start
        print(
            f"  Shard {shard_id} saved with {index.ntotal} vectors "
            f"to {index_path.name} ({shard_elapsed * 1_000_000:.1f} μs)"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"  • Total shard building took {total_elapsed * 1_000_000:.1f} μs")

    print(f"\nGenerating {TEST_VECTORS_COUNT} test vectors...")
    test_vectors = generate_random_vectors(TEST_VECTORS_COUNT, DIMS, 99999)
    np.save(test_vectors_path(), test_vectors)
    print(f"Test vectors saved to {test_vectors_path().name}")


def load_test_vectors() -> np.ndarray:
    return np.load(test_vectors_path())


def compute_ground_truth(test_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground truth by exact search on all vectors."""
    print("\nComputing ground truth (exact search on all data)...")

    # Load all vectors from all shards
    all_vectors = []
    all_ids = []

    for shard_id in range(SHARDS):
        shard_vectors = generate_random_vectors(
            VECTORS_PER_SHARD, DIMS, shard_id * 1000
        )
        start_id, _ = shard_id_range(shard_id)
        shard_ids = np.arange(start_id, start_id + VECTORS_PER_SHARD)
        all_vectors.append(shard_vectors)
        all_ids.append(shard_ids)

    all_vectors = np.vstack(all_vectors).astype(np.float32)
    all_ids = np.hstack(all_ids)

    # Build exact index
    exact_index = faiss.IndexFlatIP(DIMS)
    exact_index_with_ids = faiss.IndexIDMap(exact_index)
    exact_index_with_ids.add_with_ids(all_vectors, all_ids.astype(np.int64))

    # Search for ground truth
    gt_distances, gt_ids = exact_index_with_ids.search(test_vectors, TOP_K)

    print(f"  Ground truth computed for {len(test_vectors)} queries")
    return gt_distances, gt_ids


def calculate_recall(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """Calculate recall percentage between retrieved and ground truth IDs."""
    # Convert to sets for easy comparison
    retrieved_set = set(retrieved_ids[retrieved_ids != -1])
    ground_truth_set = set(ground_truth_ids[ground_truth_ids != -1])

    if len(ground_truth_set) == 0:
        return 0.0

    # Count how many ground truth items were retrieved
    correct = len(retrieved_set.intersection(ground_truth_set))
    recall = (correct / len(ground_truth_set)) * 100.0

    return recall


def distributed_search_with_reranking() -> None:
    print("\n=== Distributed Search (With Exact Reranking) ===")
    print(
        "Parameters: "
        f"CANDIDATES_PER_SHARD={CANDIDATES_PER_SHARD}, "
        f"nprobe={NPROBE}, NLIST={NLIST}\n"
    )

    test_vectors = load_test_vectors()
    print(f"Loaded {len(test_vectors)} test vectors")

    # Compute ground truth for recall calculation
    gt_distances, gt_ids = compute_ground_truth(test_vectors)

    shard_indexes = []
    open_start = time.perf_counter()

    for shard_id in range(SHARDS):
        shard_start = time.perf_counter()
        index = faiss.read_index(str(shard_index_path(shard_id)), faiss.IO_FLAG_MMAP)
        index.nprobe = min(index.nlist, NPROBE)
        shard_elapsed = time.perf_counter() - shard_start
        shard_indexes.append(index)
        print(
            f"  Created view of shard {shard_id} with {index.ntotal} vectors "
            f"({shard_elapsed * 1_000_000:.1f} μs)"
        )

    print(
        f"  • Total shard view creation took {(time.perf_counter() - open_start) * 1_000_000:.1f} μs"
    )

    recall_scores = []  # Track recall for each query

    for test_id, query in enumerate(test_vectors):
        print(f"\nQuery vector {test_id}:")
        query_batch = query.reshape(1, -1)

        candidate_sources: dict[int, int] = {}

        collect_start = time.perf_counter()
        for shard_id, shard in enumerate(shard_indexes):
            distances, ids = shard.search(query_batch, CANDIDATES_PER_SHARD)

            for local_id in ids[0]:
                if local_id == -1:
                    continue
                # Convert local ID to global ID
                global_id = local_to_global(local_id, shard_id)
                if global_id in candidate_sources:
                    continue
                candidate_sources[int(global_id)] = shard_id

        elapsed = time.perf_counter() - collect_start
        print(f"  Collected {len(candidate_sources)} candidates from all shards")
        print(f"    • Candidate collection took {elapsed * 1_000_000:.1f} μs")

        # Measure recall achieved by the approximate search stage alone
        candidate_ids = np.fromiter(candidate_sources.keys(), dtype=np.int64)
        candidate_recall = calculate_recall(candidate_ids, gt_ids[test_id])
        print(f"    • Candidate recall before rerank: {candidate_recall:.1f}%")

        if not candidate_sources:
            print("  No candidates retrieved; skipping rerank")
            continue

        rerank_vectors = []
        rerank_ids = []

        build_start = time.perf_counter()
        for global_key, shard_id in candidate_sources.items():
            # Convert global ID back to local ID for this shard
            local_id = global_to_local(global_key, shard_id)
            vector = shard_indexes[shard_id].reconstruct(local_id)
            rerank_vectors.append(vector)
            rerank_ids.append(global_key)

        rerank_matrix = np.stack(rerank_vectors).astype(np.float32)
        # Use flat index for exact reranking (not IVF since we want precision here)
        base_rerank_index = faiss.IndexFlatIP(DIMS)
        rerank_index = faiss.IndexIDMap(
            base_rerank_index
        )  # Wrap with IDMap to support custom IDs
        rerank_index.add_with_ids(rerank_matrix, np.asarray(rerank_ids, dtype=np.int64))
        build_elapsed = time.perf_counter() - build_start
        print(f"    • Building rerank index took {build_elapsed * 1_000_000:.1f} μs")

        search_start = time.perf_counter()
        distances, ids = rerank_index.search(query_batch, TOP_K)
        search_elapsed = time.perf_counter() - search_start
        print(f"    • Exact rerank search took {search_elapsed * 1_000_000:.1f} μs")

        # Calculate recall
        recall = calculate_recall(ids[0], gt_ids[test_id])
        recall_scores.append(recall)

        print(f"  Exact reranking top-{TOP_K} results (Recall: {recall:.1f}%):")
        for rank, (key, distance) in enumerate(zip(ids[0], distances[0]), start=1):
            if key == -1:
                continue
            shard_id = candidate_sources[int(key)]
            # Check if this result is in ground truth
            is_correct = key in gt_ids[test_id]
            correct_marker = "✓" if is_correct else " "
            print(
                f"    {rank}: {correct_marker} key={int(key)} score={float(distance):.6f} (from shard {shard_id})"
            )

    # Print summary statistics
    if recall_scores:
        avg_recall = np.mean(recall_scores)
        min_recall = np.min(recall_scores)
        max_recall = np.max(recall_scores)
        print("\n=== Recall Summary ===")
        print(f"Average Recall: {avg_recall:.1f}%")
        print(f"Min Recall: {min_recall:.1f}%")
        print(f"Max Recall: {max_recall:.1f}%")


def main() -> None:
    build_shards()
    distributed_search_with_reranking()


if __name__ == "__main__":
    main()
