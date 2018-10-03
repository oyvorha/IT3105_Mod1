from GANN import *
import WineYeastGlass.filereader as fr


def wine_class(dims=[11, 100, 100, 100, 10, 8], lrate=0.001, mbs=500, vint=250, softmax=True, optimizer="adam", error="crossentropy",
               runs=5000, bestk=1, no_of_cases=100, display_grabvars=[],
               dendrogram_layers=[], dendrogram=False, range_min=-.1, range_max=.1,
                vfrac=.1, tfrac=.1, hidden_func="relu"):
    def cfunc_wine_vectors():
        return fr.read_file("WineYeastGlass/wine.txt")
    cman = Caseman(cfunc_wine_vectors, vfrac, tfrac)
    gann_wine_vectors = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error, range_min, range_max, hidden_func)
    gann_wine_vectors.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_wine_vectors.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk,
                                     dendrogram)


def yeast_class(dims=[8, 100, 100, 10], lrate=0.001, mbs=512, vint=500, softmax=True, optimizer="adam", error="crossentropy",
                runs=10000, bestk=1, no_of_cases=10, display_grabvars=[],
                dendrogram_layers=[], dendrogram=False, range_min=-.1, range_max=.1,
                vfrac=.1, tfrac=.1, hidden_func="relu"):
    def cfunc_yeast_vectors():
        return fr.read_file("WineYeastGlass/yeast.txt")
    cman = Caseman(cfunc_yeast_vectors, vfrac, tfrac)
    gann_yeast_vectors = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error, range_min, range_max, hidden_func)
    gann_yeast_vectors.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_yeast_vectors.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram)


def glass_class(dims=[9, 100, 100, 100, 6], lrate=0.01, mbs=200, vint=500, softmax=True, optimizer="adam", error="crossentropy",
                runs=15000, bestk=1, no_of_cases=10, display_grabvars=[],
                dendrogram_layers=[], dendrogram=False, range_min=-.1, range_max=.1,
                vfrac=.1, tfrac=.1, hidden_func="relu"):
    def cfunc_glass_vectors():
        return fr.read_file("WineYeastGlass/glass.txt")

    cman = Caseman(cfunc_glass_vectors, vfrac, tfrac)
    gann_glass_vectors = Gann(dims, cman, lrate, mbs, vint, softmax, optimizer, error,range_min, range_max, hidden_func)
    gann_glass_vectors.run(runs, bestk=bestk)
    if display_grabvars or dendrogram:
        gann_glass_vectors.do_mapping(no_of_cases, display_grabvars, dendrogram_layers, bestk, dendrogram)
