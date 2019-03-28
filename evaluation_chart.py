source_file = "chart_output.txt"
target_file = "test_chart_golden.txt"


def cal_acc(source, target):
    source_file = open(source)
    target_file = open(target)

    source_lines = source_file.readlines()
    target_lines = target_file.readlines()
    tot_num = len(source_lines)
    err_num = 0
    for idx, line in enumerate(source_lines):
        if line != target_lines[idx]:
            err_num += 1
    print("Accuracy: " + str((tot_num-err_num)/tot_num))


cal_acc(source_file, target_file)
