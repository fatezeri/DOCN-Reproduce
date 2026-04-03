#!/bin/bash

PYTHON_BIN="/usr/bin/python3"
SCRIPT_PATH="$(readlink -f "$0")"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

GROUP1=(drive stare chase)
GROUP2=(rimone refuge refuge2)
ALL_DATASETS=(drive stare chase rimone refuge refuge2)

cd "$PROJECT_DIR" || {
    echo "无法进入项目目录: $PROJECT_DIR"
    exit 1
}

mkdir -p logs
mkdir -p snapshot
mkdir -p results
mkdir -p summary

# 运行前清空 logs/ 和 summary/ 中的旧文件
if [ -n "$(find logs -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "检测到 logs/ 中有旧文件，正在清空..."
    rm -rf logs/*
fi

if [ -n "$(find summary -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "检测到 summary/ 中有旧文件，正在清空..."
    rm -rf summary/*
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python 不存在或不可执行: $PYTHON_BIN"
    exit 1
fi

MASTER_LOG="logs/master_train_test_$(date +%F_%H-%M-%S).log"

echo "==================================================" | tee -a "$MASTER_LOG"
echo "两卡调度：先并行 run0/run1，再串行 run2；其余逻辑不变" | tee -a "$MASTER_LOG"
echo "项目目录: $PROJECT_DIR" | tee -a "$MASTER_LOG"
echo "项目名: $PROJECT_NAME" | tee -a "$MASTER_LOG"
echo "Python: $PYTHON_BIN" | tee -a "$MASTER_LOG"
echo "GROUP1: ${GROUP1[*]}" | tee -a "$MASTER_LOG"
echo "GROUP2: ${GROUP2[*]}" | tee -a "$MASTER_LOG"
echo "ALL_DATASETS: ${ALL_DATASETS[*]}" | tee -a "$MASTER_LOG"
echo "开始时间: $(date)" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"

run_tests_for_source() {
    local source_ds=$1
    local run_num=$2
    local gpu_id=$3
    local group_log=$4

    local targets=()

    if [[ " ${GROUP1[*]} " =~ " ${source_ds} " ]]; then
        targets=("${GROUP1[@]}")
    elif [[ " ${GROUP2[*]} " =~ " ${source_ds} " ]]; then
        targets=("${GROUP2[@]}")
    else
        echo "未知 source dataset: $source_ds" | tee -a "$group_log"
        return 1
    fi

    for target_ds in "${targets[@]}"
    do
        local test_log="logs/test_${source_ds}_to_${target_ds}_run${run_num}_$(date +%F_%H-%M-%S).log"

        echo "" | tee -a "$group_log"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | tee -a "$group_log"
        echo "开始测试: ${source_ds} -> ${target_ds}, run=${run_num}, gpu=${gpu_id}" | tee -a "$group_log"
        echo "测试日志: $test_log" | tee -a "$group_log"
        echo "执行命令: CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN predict.py $source_ds $target_ds $run_num" | tee -a "$group_log"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | tee -a "$group_log"

        CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" predict.py "$source_ds" "$target_ds" "$run_num" > "$test_log" 2>&1
        local status=$?

        if [ $status -ne 0 ]; then
            echo "测试失败: ${source_ds} -> ${target_ds}, run=${run_num}, 退出码=$status" | tee -a "$group_log"
            echo "请检查测试日志: $test_log" | tee -a "$group_log"
            return $status
        fi

        echo "测试完成: ${source_ds} -> ${target_ds}, run=${run_num}" | tee -a "$group_log"
    done

    return 0
}

run_one_group() {
    local run_num=$1
    local gpu_id=$2

    local group_log="logs/run${run_num}_master_$(date +%F_%H-%M-%S).log"

    echo "==================================================" | tee -a "$group_log"
    echo "run${run_num} 开始，GPU=${gpu_id}" | tee -a "$group_log"
    echo "开始时间: $(date)" | tee -a "$group_log"
    echo "==================================================" | tee -a "$group_log"

    for ds in "${ALL_DATASETS[@]}"
    do
        local train_log="logs/train_${ds}_run${run_num}_$(date +%F_%H-%M-%S).log"

        echo "" | tee -a "$group_log"
        echo "--------------------------------------------------" | tee -a "$group_log"
        echo "开始训练数据集: $ds" | tee -a "$group_log"
        echo "RUN_NUM: $run_num" | tee -a "$group_log"
        echo "GPU_ID: $gpu_id" | tee -a "$group_log"
        echo "开始时间: $(date)" | tee -a "$group_log"
        echo "训练日志: $train_log" | tee -a "$group_log"
        echo "执行命令: CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN train.py $ds $run_num" | tee -a "$group_log"
        echo "--------------------------------------------------" | tee -a "$group_log"

        CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" train.py "$ds" "$run_num" > "$train_log" 2>&1
        local train_status=$?

        if [ $train_status -ne 0 ]; then
            echo "训练失败: $ds run${run_num}" | tee -a "$group_log"
            echo "请检查训练日志: $train_log" | tee -a "$group_log"
            return $train_status
        fi

        echo "训练完成: $ds run${run_num}" | tee -a "$group_log"
        echo "开始组内自动测试（含同域）..." | tee -a "$group_log"

        run_tests_for_source "$ds" "$run_num" "$gpu_id" "$group_log"
        if [ $? -ne 0 ]; then
            echo "测试失败: $ds run${run_num}" | tee -a "$group_log"
            return 1
        fi

        echo "该数据集全部测试完成: $ds run${run_num}" | tee -a "$group_log"
    done

    echo "" | tee -a "$group_log"
    echo "==================================================" | tee -a "$group_log"
    echo "run${run_num} 完成" | tee -a "$group_log"
    echo "结束时间: $(date)" | tee -a "$group_log"
    echo "==================================================" | tee -a "$group_log"
    return 0
}

# -----------------------------
# 两卡调度策略
# 第一阶段：run0 -> GPU0, run1 -> GPU1 并行
# 第二阶段：run2 -> GPU0 串行
# -----------------------------

echo "启动第一阶段：run0 -> GPU0, run1 -> GPU1" | tee -a "$MASTER_LOG"

run_one_group 0 0 &
pid0=$!
echo "启动 run0 -> GPU 0, PID=$pid0" | tee -a "$MASTER_LOG"

run_one_group 1 1 &
pid1=$!
echo "启动 run1 -> GPU 1, PID=$pid1" | tee -a "$MASTER_LOG"

fail=0

wait "$pid0" || fail=1
wait "$pid1" || fail=1

echo "第一阶段结束" | tee -a "$MASTER_LOG"

echo "启动第二阶段：run2 -> GPU0" | tee -a "$MASTER_LOG"
run_one_group 2 0
phase2_status=$?
if [ $phase2_status -ne 0 ]; then
    fail=1
fi

echo "==================================================" | tee -a "$MASTER_LOG"
echo "训练与测试阶段结束，开始解析日志并生成 CSV..." | tee -a "$MASTER_LOG"

PROJECT_NAME="$PROJECT_NAME" "$PYTHON_BIN" << 'PY'
import os
import re
import csv
import math

log_dir = "logs"
summary_dir = "summary"
os.makedirs(summary_dir, exist_ok=True)
project_name = os.environ.get("PROJECT_NAME", "").strip()

pattern = re.compile(r"test_(.*?)_to_(.*?)_run(\d+)_(.*?)\.log")
f1_pattern = re.compile(r"f1score:\s*([0-9.]+)")

records = []

for fname in os.listdir(log_dir):
    m = pattern.match(fname)
    if not m:
        continue

    source, target, run, timestamp = m.groups()
    run = int(run)

    with open(os.path.join(log_dir, fname), "r", errors="ignore") as f:
        text = f.read()
        match = f1_pattern.findall(text)
        f1 = float(match[-1]) if match else None

    records.append((run, source, target, f1, fname))

# 同一个 (run, source, target) 如果有多个日志，取时间戳更靠后的那个
latest = {}
for run, source, target, f1, fname in records:
    key = (run, source, target)
    if key not in latest or fname > latest[key][4]:
        latest[key] = (run, source, target, f1, fname)

records = list(latest.values())
records.sort(key=lambda x: (x[0], x[1], x[2]))

# 导出长表
with open(os.path.join(summary_dir, "f1_long.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "source", "target", "f1", "log_file"])
    for run, source, target, f1, fname in records:
        writer.writerow([run, source, target, "" if f1 is None else f"{f1:.6f}", fname])

GROUP1 = ["drive", "stare", "chase"]
GROUP2 = ["rimone", "refuge", "refuge2"]
RUNS = [0, 1, 2]

table = {(run, source, target): f1 for run, source, target, f1, _ in records}

def calc_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def calc_std(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var)

def write_run_matrix(group, name, run):
    out_path = os.path.join(summary_dir, f"{name}_run{run}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source\\target"] + group)
        for s in group:
            row = [s]
            for t in group:
                v = table.get((run, s, t))
                row.append("" if v is None else f"{v:.3f}")
            writer.writerow(row)

def write_stat_matrix(group, name, stat_name, getter):
    out_path = os.path.join(summary_dir, f"{name}_{stat_name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source\\target"] + group)
        for s in group:
            row = [s]
            for t in group:
                v = getter(s, t)
                row.append("" if v is None else f"{v:.3f}")
            writer.writerow(row)

def write_pm_matrix(group, name):
    if project_name:
      out_name = f"{project_name}-{name}_summary.csv"
    else:
      out_name = f"{name}_summary.csv"
    out_path = os.path.join(summary_dir, out_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source\\target"] + group)
        for s in group:
            row = [s]
            for t in group:
                vals = [table.get((run, s, t)) for run in RUNS]
                vals = [v for v in vals if v is not None]
                if not vals:
                    row.append("")
                    continue
                mean = calc_mean(vals)
                std = calc_std(vals)
                row.append(f"{mean:.3f}±{std:.3f}")
            writer.writerow(row)

for run in RUNS:
    write_run_matrix(GROUP1, "group1", run)
    write_run_matrix(GROUP2, "group2", run)

def mean_getter(s, t):
    vals = [table.get((run, s, t)) for run in RUNS]
    return calc_mean(vals)

def std_getter(s, t):
    vals = [table.get((run, s, t)) for run in RUNS]
    return calc_std(vals)

write_stat_matrix(GROUP1, "group1", "mean", mean_getter)
write_stat_matrix(GROUP1, "group1", "std", std_getter)
write_pm_matrix(GROUP1, "group1")

write_stat_matrix(GROUP2, "group2", "mean", mean_getter)
write_stat_matrix(GROUP2, "group2", "std", std_getter)
write_pm_matrix(GROUP2, "group2")

print("CSV 已生成到 summary/:")
print("  - f1_long.csv")
for run in RUNS:
    print(f"  - group1_run{run}.csv")
    print(f"  - group2_run{run}.csv")
print("  - group1_mean.csv")
print("  - group1_std.csv")
if project_name:
    print(f"  - {project_name}-group1_summary.csv")
else:
    print("  - group1_summary.csv")
print("  - group2_mean.csv")
print("  - group2_std.csv")
if project_name:
    print(f"  - {project_name}-group2_summary.csv")
else:
    print("  - group2_summary.csv")
PY

PARSE_STATUS=$?
if [ $PARSE_STATUS -ne 0 ]; then
    echo "日志解析失败，请检查 logs/ 和 summary/" | tee -a "$MASTER_LOG"
else
    echo "日志解析完成，CSV 已保存到 summary/" | tee -a "$MASTER_LOG"
fi

echo "==================================================" | tee -a "$MASTER_LOG"
if [ $fail -eq 0 ]; then
    echo "全部完成（含 mean/std/mean±std 汇总）" | tee -a "$MASTER_LOG"
else
    echo "存在失败，请检查 logs/" | tee -a "$MASTER_LOG"
fi
echo "结束时间: $(date)" | tee -a "$MASTER_LOG"
echo "==================================================" | tee -a "$MASTER_LOG"

exit $fail