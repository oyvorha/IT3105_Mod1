import tflowtools as tft


def read_file(filename):
    cases = []
    file_obj = open(filename, 'r')
    if filename.split('/')[1] == "wine.txt":
        for line in file_obj.readlines():
            line_vec = line.split(';')
            input_vec = line_vec[:11]
            label = int(line_vec[-1])-1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 8)])
    if filename.split('/')[1] == "yeast.txt":
        for line in file_obj.readlines():
            line_vec = line.split(',')
            input_vec = line_vec[:8]
            label = int(line_vec[-1])-1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 10)])
    if filename.split('/')[1] == "glass.txt":
        for line in file_obj.readlines():
            line_vec = line.split(',')
            input_vec = line_vec[:9]
            label = int(line_vec[-1]) - 1
            if label > 4:
                label -= 1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 6)])
    return cases
