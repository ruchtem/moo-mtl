import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("numpy_arrays", nargs='+')
args = parser.parse_args()

for file in args.numpy_arrays:
    data = np.load(file)
    assert data.ndim == 2
    plt.plot(data[:, 0], data[:, 1], 'o', label=file)
    print("min error: {} for file {}".format(min(data[:, 1]), file))


plt.xlabel("params abs sum")
plt.ylabel("misclassification rate (test)")
plt.xlim(left=0, right=1000)
plt.ylim(bottom=0, top=0.5)
plt.legend()
plt.savefig("t.png")