# Results on multi-mnist datasets (ParetoMTL implementation)

Adam, lr 1e-4
metric: cross entropy (left range & right range)


| method          | epochs | multi-mnist test          | multi-mnist val           |
|-----------------|--------|---------------------------|---------------------------|
| single task     | 100    | 0.2832      & 0.3096      | 0.139       & 0.210       |
| a fusion early  |        |                           |                           |
| a fusion late   |        |                           |                           |
| hypernetwork    | 150    | 0.316-0.324 & 0.361-0.356 | 0.205-0.223 & 0.247-0.241 |
| paretoMTL       | 100    |                           |                           |




| method          | epochs | multi-fashion+mnist test  | multi-fashion+mnist val   |
|-----------------|--------|---------------------------|---------------------------|
| single task     | 100    |             &             |             &             |
| a fusion early  |        |                           |                           |
| a fusion late   |        |                           |                           |
| hypernetwork    | 150    |             &             |             &             |
| paretoMTL       | 100    |                           |                           |


