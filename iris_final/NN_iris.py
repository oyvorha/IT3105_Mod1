from GANN import *
import WineYeastGlass.filereader as fr


def iris_class(dims=[4, 10, 3], lrate=0.1, showint=2, mbs=10, vint=100, softmax=False):
    def cfunc_iris_vectors():
        return fr.read_file("iris.txt")
    cman = Caseman(cfunc_iris_vectors)
    gann_iris_vectors = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    print("Vectors")
    gann_iris_vectors.run(500, None, False, 1)


iris_class()

