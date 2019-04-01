import math

import numpy as np
from numba import cuda


def gpu_dedisperse(cand, device=0):
    """

    :param cand: Candidate object
    :param device: GPU id
    :return:
    """
    cuda.select_device(device)
    chan_freqs = cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32))
    cand_data_in = cuda.to_device(np.array(cand.data.T, dtype=np.uint8))
    cand_data_out = cuda.to_device(np.zeros_like(cand.data.T, dtype=np.uint8))

    @cuda.jit
    def gpu_dedisp(cand_data_in, chan_freqs, dm, cand_data_out, tsamp):
        ii, jj = cuda.grid(2)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1]:
            disp_time = int(-4148808.0 * dm * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2) / 1000 / tsamp)
            cand_data_out[ii, jj] = cand_data_in[ii, (jj + disp_time) % cand_data_in.shape[1]]

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])

    blockspergrid = (blockspergrid_x, blockspergrid_y)

    gpu_dedisp[blockspergrid, threadsperblock](cand_data_in, chan_freqs, float(cand.dm), cand_data_out,
                                               float(cand.tsamp))

    cand.dedispersed = cand_data_out.copy_to_host().T

    cuda.close()

    return cand


def gpu_dmt(cand, device=0):
    """

    :param cand: Candidate object
    :param device: GPU id
    :return:
    """
    cuda.select_device(device)
    chan_freqs = cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32))
    dm_list = cuda.to_device(np.linspace(0, 2 * cand.dm, 256, dtype=np.float32))
    dmt_return = cuda.to_device(np.zeros((256, cand.data.shape[0]), dtype=np.float32))
    cand_data_in = cuda.to_device(np.array(cand.data.T, dtype=np.uint8))

    @cuda.jit
    def gpu_dmt(cand_data_in, chan_freqs, dms, cand_data_out, tsamp):
        ii, jj, kk = cuda.grid(3)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1] and kk < dms.shape[0]:
            disp_time = int(
                -1 * 4148808.0 * dms[kk] * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2) / 1000 / tsamp)
            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii, (jj + disp_time) % cand_data_in.shape[1]])

    threadsperblock = (16, 8, 8)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(dm_list.shape[0] / threadsperblock[2])

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_dmt[blockspergrid, threadsperblock](cand_data_in, chan_freqs, dm_list, dmt_return, float(cand.tsamp))

    cand.dmt = dmt_return.copy_to_host()

    cuda.close()

    return cand
