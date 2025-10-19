#!/bin/bash
# =============================================
# 自动重复运行 main.py 以进行独立训练实验
# 每次运行后自动重命名 outcomes 文件夹。
# =============================================

# 可配置参数
RUNS=3                      # 要重复训练的次数
MAIN_SCRIPT="main.py"       # 主程序文件
OUT_DIR="outcomes"          # 模型结果文件夹
NEW_FILE="New.pth"          # 初始模型文件
PYTHON_BIN="python"         # 用哪个 Python 解释器

echo "🚀 开始自动重复训练，共 $RUNS 次。"
echo "--------------------------------------------"

for ((i=1; i<=RUNS; i++))
do
    echo ""
    echo "========== 第 $i 次训练开始 =========="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------"

    # 删除旧的checkpoint文件
    if [ -f "$NEW_FILE" ]; then
        echo "删除旧的 New.pth ..."
        rm -f "$NEW_FILE"
    fi

    # 删除旧的 outcomes
    if [ -d "$OUT_DIR" ]; then
        echo "保存上次训练结果..."
        # 补零命名，如 outcomes_01, outcomes_02
        NEW_NAME=$(printf "%s_%02d" "$OUT_DIR" "$((i+3))")
        mv "$OUT_DIR" "$NEW_NAME"
        echo "已重命名为 $NEW_NAME"
    fi

    # 运行训练脚本
    echo "正在运行 $MAIN_SCRIPT ..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN "$MAIN_SCRIPT"

    # 检查是否训练成功生成新的 outcomes 文件夹
    if [ ! -d "$OUT_DIR" ]; then
        echo "警告：未检测到新的 outcomes 文件夹，可能训练中断。"
    else
        echo "第 $i 次训练完成。"
    fi
done

echo ""
echo "所有 $RUNS 次训练结束！"
echo "--------------------------------------------"
ls -d outcomes_*