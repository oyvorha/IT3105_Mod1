from GANN import *
import iris_final.filreader as fr


def iris_class(dims=[4, 10, 3], lrate=0.01, mbs=100, vint=100, softmax=True,
               optimizer="adam", error="crossentropy", runs=4000, bestk=1, no_of_cases=10, display_grabvars=[],
               dendrogram_layers=[], dendrogram=False):
    def cfunc_iris_vectors():
        return fr.read_file("iris.txt")
    print(cfunc_iris_vectors())
    cman = Caseman(cfunc_iris_vectors)
    gann_iris_vectors = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error)
    gann_iris_vectors.run(runs, bestk=bestk)
    if dendrogram_layers or dendrogram:
        gann_iris_vectors.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram, labels=True)
