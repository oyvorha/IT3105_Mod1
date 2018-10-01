import tflowtools as tft
from GANN import *


def bit_counter(dims=[15, 500, 500, 16], lrate=0.1, showint=2, mbs=10, vint=100, softmax=True):
    def cfunc_bit_counter():
        return tft.gen_vector_count_cases(500, 15)
    cman = Caseman(cfunc_bit_counter)
    gann_bit_counter = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_bit_counter.run(500, None, False, 1)


def parity(dims=[10, 50, 2], lrate=0.1, showint=2, mbs=10, vint=100, softmax=True):
    def cfunc_parity():
        return tft.gen_all_parity_cases(10)

    print(cfunc_parity())
    cman = Caseman(cfunc_parity)
    gann_parity = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_parity.run(500, None, False, 1)


def symmetry(dims=[101, 50, 2], lrate=0.1, showint=2, mbs=10, vint=100, softmax=True):
    def cfunc_symmetry():
        cases = []
        dataset = tft.gen_symvect_dataset(100, 2000)
        for i in range(2000):
            if tft.check_vector_symmetry(dataset[i]):
                label = [1, 0]
            else:
                label = [0, 1]
            cases.append([dataset[i], label])
        return cases

    cman = Caseman(cfunc_symmetry)
    gann_symmetry = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_symmetry.run(50, None, False, 1)


def segmented_vectors(dims=[25, 500, 500, 9], lrate=0.1, showint=2, mbs=10, vint=100, softmax=True):
    def cfunc_seg_vectors():
        return tft.gen_segmented_vector_cases(25, 1000, 0, 8)

    cman = Caseman(cfunc_seg_vectors)
    gann_seg_vectors = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_seg_vectors.run(500, None, False, 1)
