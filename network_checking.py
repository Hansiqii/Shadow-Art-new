import os
import matplotlib.pyplot as plt
import numpy as np

def save_loss_plots(draw_loss, real_figure_loss):
    # 生成保存图像的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outcome_dir = os.path.join(current_dir, 'outcomes')

    # 如果不存在outcomes文件夹，创建它
    if not os.path.exists(outcome_dir):
        os.makedirs(outcome_dir)

    # 提取各个损失值
    avg_loss = [entry[0] for entry in draw_loss]
    avg_figure_loss = [entry[1] for entry in draw_loss]
    avg_regular_loss_1 = [entry[2] for entry in draw_loss]
    avg_regular_loss_2 = [entry[3] for entry in draw_loss]
    avg_regular_loss_3 = [entry[4] for entry in draw_loss]
    avg_regular_loss_6 = [entry[5] for entry in draw_loss]

    # 创建图像1
    plt.figure(figsize=(10, 6))
    epochs = [epoch + 1 for epoch in range(len(draw_loss))]

    plt.plot(epochs, avg_loss, label='Average Loss', marker='o')
    plt.plot(epochs, avg_figure_loss, label='Average Figure Loss', marker='o')
    plt.plot(epochs, avg_regular_loss_1, label='Average Regular Each Near Loss', marker='o')
    plt.plot(epochs, avg_regular_loss_2, label='Average Regular Near 0/1 Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Values over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outcome_dir, 'A_loss_plot_1.png'))
    plt.close()

    # 创建图像2
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_regular_loss_3, label='Average Regular Minimum Volume Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regular Minimum Volume Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outcome_dir, 'A_loss_plot_2.png'))
    plt.close()

    # 创建图像3
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_regular_loss_6, label='Average Regular Curvature Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regular Curvature Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outcome_dir, 'A_loss_plot_3.png'))
    plt.close()

    # 创建图像4
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, real_figure_loss, label='Average Figure Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Figure Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outcome_dir, 'A_loss_plot_4.png'))
    plt.close()

#
# # 示例的 draw_loss 数据
# draw_loss = [
#     (0.03, 0.02, 0.01, 0.04, 14.75, 13.80),
#     (0.028, 0.018, 0.009, 0.039, 14.50, 13.60),
#     # 继续添加数据...
# ]
#
# # 调用函数
# save_loss_plots(draw_loss)








