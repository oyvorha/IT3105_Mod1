class Settings():

    def __init__(self):
        self.steps = None
        self.val_interval = None
        self.best_k = None
        self.map_batch_size = None
        self.map_layers = []
        self.map_dendograms = []
        self.display_variables = False
        self.dims = []
        self.hidden_act_func = None
        self.output_act_func = None
        self.loss_func = None
        self.optimizer = None
        self.learning_rate = None
        self.init_wgt_range = []
        self.data_source = None
        self.data_source_param = []
        self.val_frac = None
        self.test_frac = None
        self.mbs = None


def read_file ():
    config = Settings()
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

        elif label.lower().rstrip() == "display variables":
            if val_is_set:
                display_var = []
                for i in range(0, len(val)-1, 2):
                    display_var.append((int(val[i]), val[i+1]))
                config.display_variables = display_var

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

        elif label.lower().rstrip() == "learning rate":
            config.learning_rate = float(val[0].rstrip())

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

        elif label.lower().rstrip() == "validation fraction":
            config.val_frac = float(val[0].rstrip())

        elif label.lower().rstrip() == "test fraction":
            config.test_frac = float(val[0].rstrip())

        elif label.lower().rstrip() == "minibatch size":
            config.mbs = int(val[0].rstrip())

        file_obj.close()

    return config
