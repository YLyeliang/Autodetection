import matplotlib.pyplot as plt


def plot():
    a = [43, 22, 18, 17.9, 17.8, 17.877, 17.966, 17.677, 17.878, 17.666, 17.555, 17.877]
    print(len(a))
    b = [54, 20, 18, 16, 13, 11.9, 10.2, 10.1, 9.77, 9.64, 8.77, 8.44]
    print(len(b))
    x = [i * 10 for i in range(12)]

    plt.plot(x, a, 'r--', label='without warmup')
    plt.plot(x, b, 'g--', label="with warmup")
    plt.xlabel('iterations')
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def parse_refinedet_txt(txt):
    """
    Each line contain iter xx || loss xx || timer
    Args:
        txt:

    Returns:

    """
    result = {}
    with open(txt, 'r') as f:
        result['iter'] = []
        result['arm_l'] = []
        result['arm_c'] = []
        result['odm_l'] = []
        result['dom_c'] = []

        while True:
            line = f.readline()
            if line is None or line is '':
                break
            iter, loss, time = line.split('||')
            loss_list = loss.strip().split(' ')
            result["iter"].append(int(iter.split(' ')[1]))
            result['arm_l'].append(int(loss_list[2]))
            result['arm_c'].append(int(loss_list[5]))
            result['odm_l'].append(int(loss_list[8]))
            result['odm_c'].append(int(loss_list[-1]))

    return result


def loss_plot(info_dict, info_dict2):
    """

    Args:
        info_dict: (dict) contains iter, arm_l, arm_c, odm_l, odm_c

    Returns:

    """
    plt.plot(info_dict['iter'], info_dict['arm_l'], 'r', label='arm_l_scratch')
    plt.plot(info_dict2['iter'], info_dict2['arm_l'], 'r--', label='arm_l_pretrained')

