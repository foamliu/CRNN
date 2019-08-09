import matplotlib.pyplot as plt

if __name__ == '__main__':
    k = 1
    warmup_steps = 50000
    init_lr = 0.2

    lr_list = []

    for step_num in range(1, 2000000):
        # print(step_num)
        lr_1 = k * init_lr * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))

        # print(lr_1)
        # print(lr_2)
        lr_list.append(lr_1)

        # if step_num > 20:
        #     break

    print(lr_list[:100])
    print(lr_list[-100:])

    plt.plot(lr_list)
    plt.show()
