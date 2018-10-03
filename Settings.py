class Settings():
    import tensorflow as tf
    import numpy as np
    import math
    import matplotlib.pyplot as PLT
    import tflowtools as TFT
    import random


    def __init__(self):
        self.steps = None
        self.val_interval = None
        self.best_k = None
        self.map_batch_size = None
        self.map_layers = []
        self.map_dendograms = []
        self.display_weights = False
        self.display_biases = False
        self.dims = []
        self.hidden_act_func = None
        self.output_act_func = None
        self.loss_func = None
        self.optimizer = None
        self.optimizer_param = []
        self.learning_rate = None
        self.wgt_init_met = None
        self.init_wgt_range = []
        self.data_source = None
        self.data_source_param = []
        self.case_frac = None
        self.val_frac = None
        self.test_frac = None
        self.mbs = None


def read_file ():
    config = Settings()
    cases = []
    file_obj = open("input", 'r')
    for line in file_obj.readlines():
        line_vec = line.split(':')
        label = line_vec[0]
        val_is_set = False

        if line_vec[1] != "\n":
            val = line_vec[1:]
            val_is_set = True

        if label.lower().rstrip() == "steps":
            config.steps = int(val[0].rstrip())

        elif label.lower().rstrip() == "validation interval":
            config.val_interval = int(val[0].rstrip())

        elif label.lower().rstrip() == "best k":
            config.best_k = int(val[0].rstrip())

        elif label.lower().rstrip() == "map batch size":
            config.map_batch_size = int(val[0].rstrip())

        elif label.lower().rstrip() == "map layers":
            if val_is_set:
                for v in val:
                    config.map_layers.append(int(v))

        elif label.lower().rstrip() == "map dendograms":
            if val_is_set:
                for v in val:
                    config.map_dendograms.append(int(v))

        elif label.lower().rstrip() == "display weights":
            if val_is_set:
                config.display_weights = True

        elif label.lower().rstrip() == "display biases":
            if val_is_set:
                config.display_biases = True

        elif label.lower().rstrip() == "network dimensions":
            if val_is_set:
                for v in val:
                    config.dims.append(int(v))

        elif label.lower().rstrip() == "hidden activation function":
            config.hidden_act_func = val[0].strip()

        elif label.lower().rstrip() == "output activation function":
            config.output_act_func = val[0].strip()

        elif label.lower().rstrip() == "loss function":
            config.loss_func = val[0].strip()

        elif label.lower().rstrip() == "optimizer":
            config.optimizer = val[0].strip()

        elif label.lower().rstrip() == "optimizer parameters":
            if val_is_set:
                for v in val:
                    config.optimizer_param.append(float(v))

        elif label.lower().rstrip() == "learning rate":
            config.learning_rate = float(val[0].rstrip())

        elif label.lower().rstrip() == "weight initializing method":
            config.wgt_init_met = val[0].strip()

        elif label.lower().rstrip() == "initial weight range":
            if val_is_set:
                for v in val:
                    config.init_wgt_range.append(float(v))

        elif label.lower().rstrip() == "data source":
            config.data_source = val[0].strip()

        elif label.lower().rstrip() == "data source parameters":
            if val_is_set:
                for v in val:
                    config.data_source_param.append(int(v))

        elif label.lower().rstrip() == "case fraction":
            config.case_frac = int(val[0].rstrip())

        elif label.lower().rstrip() == "validation fraction":
            config.val_frac = float(val[0].rstrip())

        elif label.lower().rstrip() == "test fraction":
            config.test_frac = float(val[0].rstrip())

        elif label.lower().rstrip() == "minibatch size":
            config.mbs = int(val[0].rstrip())

        file_obj.close()

    return config

read_file()