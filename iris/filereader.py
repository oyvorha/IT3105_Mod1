import tflowtools as tft


def read_file(filename):
    cases = []
    file_obj = open(filename, 'r')

    for line in file_obj.readlines():
        line_vec = line.split(',')
        input_vec = line_vec[:4]
        label = flower_to_int(line_vec[-1])
        cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 3)])


def flower_to_int (label):
    if (label == "Iris-setosa"):
        return 0
    if (label == "Iris-versicolor"):
        return 1
    return 2

