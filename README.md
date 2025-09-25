```
Building 3 shards with 1000000 vectors each NLIST=10000...
  Shard 0 saved with 1000000 vectors to shard_0.faiss (51928556.6 μs)
  Shard 1 saved with 1000000 vectors to shard_1.faiss (51789118.5 μs)
  Shard 2 saved with 1000000 vectors to shard_2.faiss (52677617.7 μs)
  • Total shard building took 156395403.2 μs

Generating 10 test vectors...
Test vectors saved to test_vectors.npy

=== Distributed Search (With Exact Reranking) ===

Loaded 10 test vectors
  Created view of shard 0 with 1000000 vectors (2436.3 μs)
  Created view of shard 1 with 1000000 vectors (2348.5 μs)
  Created view of shard 2 with 1000000 vectors (2098.2 μs)
  • Total shard view creation took 6932.6 μs

Query vector 0:
  Collected 60 candidates from all shards
    • Candidate collection took 794.8 μs
    • Building rerank index took 123.3 μs
    • Exact rerank search took 75.0 μs
  Exact reranking top-5 results:
    1: key=265925 score=40.129772 (from shard 0)
    2: key=851124 score=39.606647 (from shard 0)
    3: key=2753577 score=39.214424 (from shard 2)
    4: key=784161 score=39.053589 (from shard 0)
    5: key=1344939 score=38.658588 (from shard 1)

Query vector 1:
  Collected 60 candidates from all shards
    • Candidate collection took 968.1 μs
    • Building rerank index took 108.1 μs
    • Exact rerank search took 77.7 μs
  Exact reranking top-5 results:
    1: key=2299138 score=41.830601 (from shard 2)
    2: key=1201377 score=41.466362 (from shard 1)
    3: key=322663 score=40.990742 (from shard 0)
    4: key=1198279 score=40.985798 (from shard 1)
    5: key=1139799 score=40.836918 (from shard 1)

Query vector 2:
  Collected 60 candidates from all shards
    • Candidate collection took 801.5 μs
    • Building rerank index took 88.5 μs
    • Exact rerank search took 79.4 μs
  Exact reranking top-5 results:
    1: key=2659901 score=36.999119 (from shard 2)
    2: key=853407 score=36.654343 (from shard 0)
    3: key=1933339 score=36.516800 (from shard 1)
    4: key=195580 score=36.454132 (from shard 0)
    5: key=942140 score=36.304634 (from shard 0)

Query vector 3:
  Collected 60 candidates from all shards
    • Candidate collection took 714.8 μs
    • Building rerank index took 74.0 μs
    • Exact rerank search took 90.0 μs
  Exact reranking top-5 results:
    1: key=142789 score=41.799263 (from shard 0)
    2: key=2064855 score=39.900246 (from shard 2)
    3: key=1851780 score=39.533398 (from shard 1)
    4: key=989701 score=38.981575 (from shard 0)
    5: key=1470561 score=38.681992 (from shard 1)

Query vector 4:
  Collected 60 candidates from all shards
    • Candidate collection took 726.5 μs
    • Building rerank index took 78.4 μs
    • Exact rerank search took 73.2 μs
  Exact reranking top-5 results:
    1: key=1886205 score=41.167740 (from shard 1)
    2: key=2449007 score=40.690819 (from shard 2)
    3: key=1867019 score=40.524452 (from shard 1)
    4: key=1087692 score=40.050480 (from shard 1)
    5: key=983766 score=40.005878 (from shard 0)

Query vector 5:
  Collected 60 candidates from all shards
    • Candidate collection took 747.2 μs
    • Building rerank index took 75.8 μs
    • Exact rerank search took 70.0 μs
  Exact reranking top-5 results:
    1: key=121183 score=41.879295 (from shard 0)
    2: key=2147978 score=41.391174 (from shard 2)
    3: key=2758418 score=41.157665 (from shard 2)
    4: key=2686500 score=40.942738 (from shard 2)
    5: key=1885993 score=40.509560 (from shard 1)

Query vector 6:
  Collected 60 candidates from all shards
    • Candidate collection took 697.1 μs
    • Building rerank index took 80.7 μs
    • Exact rerank search took 58.8 μs
  Exact reranking top-5 results:
    1: key=162889 score=43.395370 (from shard 0)
    2: key=1806976 score=43.255280 (from shard 1)
    3: key=425321 score=42.516037 (from shard 0)
    4: key=2727692 score=42.505157 (from shard 2)
    5: key=82620 score=42.365463 (from shard 0)

Query vector 7:
  Collected 60 candidates from all shards
    • Candidate collection took 669.4 μs
    • Building rerank index took 73.5 μs
    • Exact rerank search took 63.7 μs
  Exact reranking top-5 results:
    1: key=1325085 score=37.009781 (from shard 1)
    2: key=1727998 score=36.864914 (from shard 1)
    3: key=1020448 score=36.630257 (from shard 1)
    4: key=951354 score=36.621639 (from shard 0)
    5: key=222734 score=36.613312 (from shard 0)

Query vector 8:
  Collected 60 candidates from all shards
    • Candidate collection took 882.8 μs
    • Building rerank index took 123.3 μs
    • Exact rerank search took 73.1 μs
  Exact reranking top-5 results:
    1: key=181902 score=39.457809 (from shard 0)
    2: key=2211691 score=39.249290 (from shard 2)
    3: key=941178 score=39.203476 (from shard 0)
    4: key=1756800 score=38.927593 (from shard 1)
    5: key=392258 score=38.872280 (from shard 0)

Query vector 9:
  Collected 60 candidates from all shards
    • Candidate collection took 887.5 μs
    • Building rerank index took 94.2 μs
    • Exact rerank search took 69.3 μs
  Exact reranking top-5 results:
    1: key=1670744 score=38.478264 (from shard 1)
    2: key=1545766 score=38.258919 (from shard 1)
    3: key=1743286 score=37.960457 (from shard 1)
    4: key=360034 score=37.783092 (from shard 0)
    5: key=2973990 score=37.515247 (from shard 2)
```
