import os
import datetime
import numpy as np
import tensorflow as tf

## These lines ensure any existing tf sessions are closed.
try:
    tf.Session().close()
except:
    pass

## Numpy print options
np.set_printoptions(precision=3)

## Directories

# path to yml files
experiment_directory = "experiments/"

# path to cpkl files
data_directory = "../results_create_files_for_protein_base/"

# path to output directory
output_directory = "../results_final/results_protein_base/"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# A slightly fancy printing method
def printt(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))
