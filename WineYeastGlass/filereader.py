import tflowtools as tft


def read_file(filename):
    cases = []
    file_obj = open(filename, 'r')
    if filename.split('.')[0] == "wine":
        for line in file_obj.readlines():
            line_vec = line.split(';')
            input_vec = line_vec[:10]
            label = int(line_vec[-1])-1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 6)])
    if filename.split('.')[0] == "yeast":
        for line in file_obj.readlines():
            line_vec = line.split(',')
            input_vec = line_vec[:7]
            label = int(line_vec[-1])-1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 10)])
    if filename.split('.')[0] == "glass":
        for line in file_obj.readlines():
            line_vec = line.split(',')
            input_vec = line_vec[:8]
            label = int(line_vec[-1]) - 1
            if label > 4:
                label -= 1
            cases.append([list(map(float, input_vec)), tft.int_to_one_hot(label, 6)])
    return cases
