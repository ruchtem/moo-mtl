# Results on multi-mnist (ParetoMTL implementation)

100 epochs, lr 1e-4
metric: cross entropy (left range & right range)


| method          | multi-mnist test | multi-mnist val  |
|-----------------|------------------|------------------|
| single task     | 0.2832 & 0.3096  | 0.139  & 0.2102  |
| a fusion early  |                  |                  |
| a fusion late   |                  |                  |
| hypernetwork    |                  |                  |
| paretoMTL       |                  |                  |

