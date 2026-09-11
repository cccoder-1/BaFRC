#!/usr/bin/env bash
# BaFRC — FewRel 训练 / 测试脚本
# 用法：直接编辑下方参数后执行
#   bash run_fewrel.sh
# 或
#   chmod +x run_fewrel.sh && ./run_fewrel.sh

set -euo pipefail

# =============================================================================
# 环境与路径（按需修改）
# =============================================================================
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# 数据路径
# =============================================================================
ROOT="${ROOT:-./data/fewrel}"
TRAIN="train_wiki"
VAL="val_wiki"
TEST="test_wiki"
PID2NAME="description_pool"
PRETRAIN_CKPT="bert-base-uncased"

# =============================================================================
# Few-shot 设置
# =============================================================================
N=5
K=1
Q=1
NA_RATE=5          # 训练使用 50% NOTA；测试会自动覆盖 15% / 30% / 50%
SEED=5

# =============================================================================
# 训练超参
# =============================================================================
BATCH_SIZE=2
TRAIN_ITER=30000
VAL_ITER=1000
TEST_ITER=10000
VAL_STEP=1000
GRAD_ITER=1
LR=2e-5
WEIGHT_DECAY=1e-5
MAX_LENGTH=128
HIDDEN_SIZE=768

# =============================================================================
# BaFRC 模型超参
# =============================================================================
BAFRC_MARGIN=0.15
BAFRC_CLS_TEMP=10.0
BAFRC_GAMMA=3.0
RADIUS_QUANTILE=0.1
RADIUS_REG=0.1
RADIUS_BLEND_RHO=1.0              # FewRel: 训练使用纯 query-based radius
BAFRC_DIST_TYPE="euclidean"          # euclidean | cosine
USE_QUERY_IN_RADIUS="true"         # true | false
USE_DESC_NEG="true"                # true | false
USE_QUERY_IN_BOUNDARY="false"      # true | false
RADIUS_UPDATE_INTERVAL=100
USE_STD_DESC="true"                # true | false
BAFRC_LOSS_POS="true"                # true | false
BAFRC_LOSS_NEG="true"                # true | false
BAFRC_LOSS_RADIUS_REG="true"         # true | false

# =============================================================================
# 运行模式（三选一，把需要的设为 1，其余设为 0）
# =============================================================================
MODE_TRAIN=1                       # 1=训练并在结束后自动测试
MODE_ONLY_TEST=0                   # 1=仅测试（需设置 LOAD_CKPT）
MODE_TEST_ONLINE=0                 # 1=在线提交测试（需 LOAD_CKPT + TEST_INPUT）

# 测试 / 在线提交时使用的 checkpoint（仅 MODE_ONLY_TEST 或 MODE_TEST_ONLINE 时需要）
LOAD_CKPT=""
SAVE_CKPT=""                       # 留空则自动命名保存到 checkpoint/

# 在线提交专用（MODE_TEST_ONLINE=1 时填写）
TEST_INPUT="test_wiki_input-5-1-0.15"
TEST_OUTPUT="pred-5-1.json"

# 可选：仅加载 encoder 权重（一般留空）
ENCODER_CKPT=""

# =============================================================================
# 日志（留空则只输出到终端）
# =============================================================================
LOG_FILE=""                        # 例如：log/fewrel-5-1-seed5.log

# =============================================================================
# 组装命令（一般无需修改）
# =============================================================================
CMD=(
  "${PYTHON}" train_fewrel.py
  --root "${ROOT}"
  --train "${TRAIN}"
  --val "${VAL}"
  --test "${TEST}"
  --pid2name "${PID2NAME}"
  --pretrain_ckpt "${PRETRAIN_CKPT}"
  --N "${N}"
  --K "${K}"
  --Q "${Q}"
  --na_rate "${NA_RATE}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --train_iter "${TRAIN_ITER}"
  --val_iter "${VAL_ITER}"
  --test_iter "${TEST_ITER}"
  --val_step "${VAL_STEP}"
  --grad_iter "${GRAD_ITER}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_length "${MAX_LENGTH}"
  --hidden_size "${HIDDEN_SIZE}"
  --bafrc_margin "${BAFRC_MARGIN}"
  --bafrc_cls_temp "${BAFRC_CLS_TEMP}"
  --bafrc_gamma "${BAFRC_GAMMA}"
  --radius_quantile "${RADIUS_QUANTILE}"
  --radius_reg "${RADIUS_REG}"
  --radius_blend_rho "${RADIUS_BLEND_RHO}"
  --bafrc_dist_type "${BAFRC_DIST_TYPE}"
  --use_query_in_radius "${USE_QUERY_IN_RADIUS}"
  --use_desc_neg "${USE_DESC_NEG}"
  --use_query_in_boundary "${USE_QUERY_IN_BOUNDARY}"
  --radius_update_interval "${RADIUS_UPDATE_INTERVAL}"
  --use_std_desc "${USE_STD_DESC}"
  --bafrc_loss_pos "${BAFRC_LOSS_POS}"
  --bafrc_loss_neg "${BAFRC_LOSS_NEG}"
  --bafrc_loss_radius_reg "${BAFRC_LOSS_RADIUS_REG}"
)

if [[ -n "${SAVE_CKPT}" ]]; then
  CMD+=(--save_ckpt "${SAVE_CKPT}")
fi

if [[ -n "${ENCODER_CKPT}" ]]; then
  CMD+=(--encoder_ckpt "${ENCODER_CKPT}")
fi

if [[ "${MODE_TRAIN}" -eq 1 ]]; then
  :
elif [[ "${MODE_ONLY_TEST}" -eq 1 ]]; then
  if [[ -z "${LOAD_CKPT}" ]]; then
    echo "[ERROR] MODE_ONLY_TEST=1 时必须设置 LOAD_CKPT" >&2
    exit 1
  fi
  CMD+=(--only_test --load_ckpt "${LOAD_CKPT}")
elif [[ "${MODE_TEST_ONLINE}" -eq 1 ]]; then
  if [[ -z "${LOAD_CKPT}" ]]; then
    echo "[ERROR] MODE_TEST_ONLINE=1 时必须设置 LOAD_CKPT" >&2
    exit 1
  fi
  CMD+=(--only_test --test_online --load_ckpt "${LOAD_CKPT}")
  CMD+=(--test_input "${TEST_INPUT}" --test_output "${TEST_OUTPUT}")
else
  echo "[ERROR] 请将 MODE_TRAIN / MODE_ONLY_TEST / MODE_TEST_ONLINE 之一设为 1" >&2
  exit 1
fi

echo "BaFRC FewRel"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  "${CMD[@]}"
fi
