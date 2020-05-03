import sys
sys.path.append('..')

import matplotlib.pyplot as plt

from spo2evaluation.preprocessing import data_loader_pandas

if __name__ == "__main__":
    dataset = data_loader_pandas.Spo2DatasetPandas('test_iphone')
    dataset.print_pgg(0)
    dataset.print_pgg(1)
    dataset.print_pgg(2)

    plt.show()
