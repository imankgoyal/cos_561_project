#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

def get_tm_mat(path_to_data):
    for root, dirs, files in os.walk(path_to_data, topdown=False):
        traffic_matrices = []
        for name in sorted(files):
            # Make sure the file is a Traffic Matrix
            if '.dat' in name:
                with open(os.path.join(root, name)) as f:
                    file_contents = f.read()
                    mat_start = 13 # Matrix elements start at newline 13
                    mat_rows = file_contents.splitlines()[mat_start:]
                    # Matrix elements are currently strings, so need to fix that
                    tm = [float(ele) for mat_row in mat_rows
                                         for ele in mat_row.split(sep=",")]
                    traffic_matrices.append(np.reshape(tm, (12, 12)))
            else:
                continue
    return np.array(traffic_matrices)

if __name__ == "__main__":
    path_to_data = "../../data/raw/Abilene/2004/Measured/"
    np.save("traffic_mats.npy", get_tm_mat(path_to_data))
