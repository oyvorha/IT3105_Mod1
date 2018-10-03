import tflowtools as tft
from GANN import *


def bit_counter(dims=[15, 10, 10, 16], lrate=0.01, mbs=500, vint=300, softmax=True, optimizer="adam", error="crossentropy",
                runs=8000, bestk=1, no_of_cases=10, display_grabvars=[],
                dendrogram_layers=[], dendrogram=False):
    def cfunc_bit_counter():
        return tft.gen_vector_count_cases(500, 15)

    cman = Caseman(cfunc_bit_counter)
    gann_bit_counter = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error)
    gann_bit_counter.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_bit_counter.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram)


def parity(dims=[10, 10, 2], lrate=0.01, mbs=100, vint=100, softmax=True, optimizer="adam", error="crossentropy",
        runs=5000, bestk=1, no_of_cases=10, display_grabvars=[], dendrogram_layers=[], dendrogram=False):
    def cfunc_parity():
        return tft.gen_all_parity_cases(10)

    cman = Caseman(cfunc_parity)
    gann_parity = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error)
    gann_parity.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_parity.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram)


def symmetry(dims=[101, 10, 2], lrate=0.001, mbs=100, vint=100, softmax=True, optimizer="adam", error="crossentropy",
             runs=5000, bestk=1, no_of_cases=10, display_grabvars=[],
             dendrogram_layers=[], dendrogram=False):
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
    print(cfunc_symmetry())
    cman = Caseman(cfunc_symmetry)
    gann_symmetry = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error)
    gann_symmetry.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_symmetry.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram, labels=True)


def segmented_vectors(dims=[25, 10, 10, 9], lrate=0.01, mbs=100, vint=500, softmax=True, optimizer="adam", error="crossentropy",
                      runs=8000, bestk=1, no_of_cases=10, display_grabvars=[],
                      dendrogram_layers=[], dendrogram=False):
    def cfunc_seg_vectors():
        return tft.gen_segmented_vector_cases(25, 1000, 0, 8)

    cman = Caseman(cfunc_seg_vectors)
    gann_seg_vectors = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error)
    gann_seg_vectors.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_seg_vectors.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram)
