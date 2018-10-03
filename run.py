import Settings as set
import GeneratedCases.NN_gen_cases as gen
import WineYeastGlass.NN_wine_yeast_glass as wyg
import mnist.NN_mnist as mnist
import iris_final.NN_iris as iris

base_path = "config_files/"
input_map = {1: "bit_counter.txt", 2: "parity.txt", 3: "symmetry.txt", 4: "segmented_vectors",
             5: "wine.txt", 6: "yeast.txt", 7: "glass.txt", 8: "mnist.txt", 9: "glass.txt"}


def run_NN(filename):
    config = set.read_file(base_path + filename)
    if filename == "bit_counter.txt":
        gen.bit_counter(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func, config.data_source_param)
    elif filename == "parity.txt":
        gen.parity(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func, config.data_source_param)
    elif filename == "symmetry.txt":
        gen.symmetry(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                   config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                   config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                   config.val_frac, config.test_frac, config.hidden_act_func, config.data_source_param)
    elif filename == "segmented_vectors.txt":
        gen.segmented_vectors(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                   config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                   config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                   config.val_frac, config.test_frac, config.hidden_act_func, config.data_source_param)
    elif filename == "wine.txt":
        wyg.wine_class(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func)
    elif filename == "yeast.txt":
        wyg.yeast_class(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func)
    elif filename == "glass.txt":
        wyg.glass_class(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func)
    elif filename == "mnist.txt":
        mnist.NN_mnist(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func)
    elif filename == "iris.txt":
        iris.iris_class(config.dims, config.learning_rate, config.mbs, config.val_interval, config.softmax, config.optimizer,
                        config.loss_func, config.steps, config.best_k, config.map_batch_size, config.display_variables,
                        config.map_dendograms, config.dendograms, config.init_wgt_range[0], config.init_wgt_range[1],
                        config.val_frac, config.test_frac, config.hidden_act_func)


def pretty_print():
    print("--------------menu-----------------")
    print("0: quit")
    print("1: bit counter")
    print("2: parity")
    print("3: symmetry")
    print("4: segmented vectors")
    print("5: wine")
    print("6: yeast")
    print("7: glass")
    print("8: mnist")
    print("9: iris (hackers choice)")
    print("10: edit settings")


def main():
    choice = None
    while choice != "0":
        pretty_print()
        choice = input()
        if int(choice) in input_map.keys():
            run_NN(input_map[int(choice)])


if __name__ == '__main__':
    main()
