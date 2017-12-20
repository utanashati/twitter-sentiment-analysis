def txt_to_list_train(path):
    list_ = []
    with open(path) as f:
        for line in f:
            list_.append(line[:-1])
            
    return list_

def txt_to_list_test(path):
    list_ = []
    with open(path) as f:
        for line in f:
            line = line[:-1]
            list_.append(line.split(',', 1))
            
    return list_