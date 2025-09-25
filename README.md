```
Building 3 shards with 3333 vectors each...
  Shard 0 saved with 3333 vectors to shard_0.faiss (9012.2 μs)
  Shard 1 saved with 3333 vectors to shard_1.faiss (1354.3 μs)
  Shard 2 saved with 3333 vectors to shard_2.faiss (1315.3 μs)
  • Total shard building took 11715.3 μs

Generating 10 test vectors...
Test vectors saved to test_vectors.npy

=== Distributed Search (With Exact Reranking) ===

Loaded 10 test vectors
  Created view of shard 0 with 3333 vectors (191.5 μs)
  Created view of shard 1 with 3333 vectors (173.5 μs)
  Created view of shard 2 with 3333 vectors (253.5 μs)
  • Total shard view creation took 639.3 μs

Query vector 0:
  Collected 60 candidates from all shards
    • Candidate collection took 697.0 μs
    • Building rerank index took 93.0 μs
    • Exact rerank search took 253.2 μs
  Exact reranking top-5 results:
    1: key=2988 score=39.526100 (from shard 0)
    2: key=2528 score=38.708855 (from shard 0)
    3: key=3541 score=38.696709 (from shard 1)
    4: key=9825 score=38.359306 (from shard 2)
    5: key=5358 score=38.321899 (from shard 1)

Query vector 1:
  Collected 60 candidates from all shards
    • Candidate collection took 169.6 μs
    • Building rerank index took 66.3 μs
    • Exact rerank search took 69.7 μs
  Exact reranking top-5 results:
    1: key=7753 score=41.539986 (from shard 2)
    2: key=4739 score=41.306583 (from shard 1)
    3: key=3921 score=41.184624 (from shard 1)
    4: key=6966 score=40.885605 (from shard 2)
    5: key=2988 score=40.631744 (from shard 0)

Query vector 2:
  Collected 60 candidates from all shards
    • Candidate collection took 122.7 μs
    • Building rerank index took 84.8 μs
    • Exact rerank search took 70.9 μs
  Exact reranking top-5 results:
    1: key=2988 score=37.879204 (from shard 0)
    2: key=3541 score=36.011143 (from shard 1)
    3: key=4348 score=35.786697 (from shard 1)
    4: key=9587 score=35.715832 (from shard 2)
    5: key=9825 score=35.641388 (from shard 2)

Query vector 3:
  Collected 60 candidates from all shards
    • Candidate collection took 130.3 μs
    • Building rerank index took 106.7 μs
    • Exact rerank search took 59.1 μs
  Exact reranking top-5 results:
    1: key=9825 score=38.427540 (from shard 2)
    2: key=9156 score=38.276466 (from shard 2)
    3: key=6366 score=38.241028 (from shard 1)
    4: key=3541 score=38.234394 (from shard 1)
    5: key=4222 score=37.915371 (from shard 1)

Query vector 4:
  Collected 60 candidates from all shards
    • Candidate collection took 141.0 μs
    • Building rerank index took 78.1 μs
    • Exact rerank search took 62.5 μs
  Exact reranking top-5 results:
    1: key=2988 score=41.595490 (from shard 0)
    2: key=9390 score=40.699726 (from shard 2)
    3: key=6685 score=39.957504 (from shard 2)
    4: key=4069 score=39.503483 (from shard 1)
    5: key=3921 score=39.405045 (from shard 1)

Query vector 5:
  Collected 60 candidates from all shards
    • Candidate collection took 197.6 μs
    • Building rerank index took 85.8 μs
    • Exact rerank search took 60.7 μs
  Exact reranking top-5 results:
    1: key=8756 score=40.557365 (from shard 2)
    2: key=6966 score=40.211411 (from shard 2)
    3: key=3508 score=40.205833 (from shard 1)
    4: key=9825 score=40.142139 (from shard 2)
    5: key=3921 score=40.117085 (from shard 1)

Query vector 6:
  Collected 60 candidates from all shards
    • Candidate collection took 128.5 μs
    • Building rerank index took 88.0 μs
    • Exact rerank search took 56.8 μs
  Exact reranking top-5 results:
    1: key=3661 score=42.579983 (from shard 1)
    2: key=3541 score=42.562492 (from shard 1)
    3: key=2895 score=42.400383 (from shard 0)
    4: key=3508 score=42.385094 (from shard 1)
    5: key=4919 score=42.338757 (from shard 1)

Query vector 7:
  Collected 60 candidates from all shards
    • Candidate collection took 114.0 μs
    • Building rerank index took 55.8 μs
    • Exact rerank search took 48.5 μs
  Exact reranking top-5 results:
    1: key=7753 score=37.071918 (from shard 2)
    2: key=1834 score=36.742409 (from shard 0)
    3: key=3541 score=36.686245 (from shard 1)
    4: key=2988 score=36.451599 (from shard 0)
    5: key=9825 score=36.399124 (from shard 2)

Query vector 8:
  Collected 60 candidates from all shards
    • Candidate collection took 111.9 μs
    • Building rerank index took 53.2 μs
    • Exact rerank search took 41.4 μs
  Exact reranking top-5 results:
    1: key=6966 score=38.814911 (from shard 2)
    2: key=2988 score=38.479595 (from shard 0)
    3: key=6719 score=38.404163 (from shard 2)
    4: key=824 score=38.148842 (from shard 0)
    5: key=1383 score=38.077290 (from shard 0)

Query vector 9:
  Collected 60 candidates from all shards
    • Candidate collection took 111.7 μs
    • Building rerank index took 50.2 μs
    • Exact rerank search took 54.0 μs
  Exact reranking top-5 results:
    1: key=2988 score=37.296005 (from shard 0)
    2: key=3921 score=37.205635 (from shard 1)
    3: key=7938 score=37.001003 (from shard 2)
    4: key=6366 score=36.929882 (from shard 1)
    5: key=8753 score=36.919910 (from shard 2)
```
# faiss_testing
