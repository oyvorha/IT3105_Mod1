from GANN import *
import WineYeastGlass.filereader as fr


def wine_class(dims=[10, 500, 6], lrate=0.1, showint=2, mbs=10, vint=100, softmax=False):
    def cfunc_wine_vectors():
        return fr.read_file("wine.txt")
    cman = Caseman(cfunc_wine_vectors)
    gann_wine_vectors = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_wine_vectors.run(500, None, False, 1)


def yeast_class(dims=[7, 500, 500, 10], lrate=0.1, showint=2, mbs=10, vint=100, softmax=False):
    def cfunc_yeast_vectors():
        return fr.read_file("yeast.txt")

    cman = Caseman(cfunc_yeast_vectors)
    gann_yeast_vectors = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_yeast_vectors.run(10, None, False, 1)


def glass_class(dims=[8, 100, 6], lrate=0.1, showint=2, mbs=10, vint=100, softmax=True):
    def cfunc_glass_vectors():
        return fr.read_file("glass.txt")

    cman = Caseman(cfunc_glass_vectors)
    gann_glass_vectors = Gann(dims, cman, lrate, showint, mbs, vint, softmax)
    gann_glass_vectors.run(500, None, False, 1)


wine_class()
