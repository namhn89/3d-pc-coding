import os
import subprocess

import numpy as np


def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv6.
    You can download and install TMC13 from
    http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
    """

    subp = subprocess.Popen(
        "tools/tmc3"
        + " --mode=0"
        + " --positionQuantizationScale=1"
        + " --trisoup_node_size_log2=0"
        + " --ctxOccupancyReductionFactor=3"
        + " --neighbourAvailBoundaryLog2=8"
        + " --intra_pred_max_node_size_log2=6"
        + " --inferredDirectCodingMode=0"
        + " --uncompressedDataPath="
        + filedir
        + " --compressedStreamPath="
        + bin_dir,
        shell=True,
        stdout=subprocess.PIPE,
    )
    c = subp.stdout.readline()
    while c:
        if show:
            print(c)
        c = subp.stdout.readline()

    return


def gpcc_decode(bin_dir, rec_dir, show=False):
    subp = subprocess.Popen(
        "myutils/tmc3"
        + " --mode=1"
        + " --compressedStreamPath="
        + bin_dir
        + " --reconstructedDataPath="
        + rec_dir,
        shell=True,
        stdout=subprocess.PIPE,
    )
    c = subp.stdout.readline()
    while c:
        if show:
            print(c)
        c = subp.stdout.readline()

    return
