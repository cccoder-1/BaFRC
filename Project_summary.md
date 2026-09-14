# BaFRC 项目结构总结

## 1. 项目概览

BaFRC 是一个面向**开放集小样本关系分类**（Open-set Few-shot Relation Classification）的 PyTorch 实验仓库，主要覆盖两个数据集：

- **FewRel**：训练时在线随机构造 N-way K-shot episode，并通过额外采样非目标关系样本构造 NOTA（None of the Above）查询。
- **FS-TACRED**：读取已经预采样好的 episode，查询关系不属于当前 N 个目标类时标记为 NOTA。

项目以 `bert-base-uncased` 为默认文本编码器，核心模型是 `BaFRC`；FewRel 入口还提供 `RoFRC` 对比模型。整体处理链路如下：

```text
JSON 数据 / 关系描述池
        ↓
Episode Dataset + DataLoader
        ↓
BERT 实体感知句子编码 / 关系描述编码
        ↓
BaFRC（或 FewRel 上的 RoFRC）
        ↓
训练框架：优化、验证、checkpoint 选择
        ↓
ACC / Macro-F1 / NOTA 指标 / Target Micro 指标
```

仓库只包含代码、数据放置说明和关系描述池，不包含原始 FewRel/FS-TACRED 数据、预训练 BERT 权重或训练好的 checkpoint。

## 2. 顶层目录树

```text
BaFRC/
├── README.md                         # 安装、数据准备和基线运行简介
├── requirements.txt                 # Python 依赖
├── train_fewrel.py                  # FewRel 训练、离线评估、在线提交入口
├── train_tacred.py                  # FS-TACRED 训练、评估、边界校准入口
├── dataset/
│   ├── __init__.py
│   ├── FewRelDataset.py             # FewRel 随机 episode 数据集
│   ├── FewRelOnlineDataset.py       # FewRel CodaLab/在线测试数据集
│   └── FSTacredEpisodeDataset.py    # FS-TACRED 预采样 episode 数据集
├── encoder/
│   ├── __init__.py
│   └── BertEncoder.py               # BERT 编码器及两类 tokenizer
├── models/
│   ├── __init__.py
│   ├── bafrc.py                     # BaFRC 主模型
│   └── RoFRC.py                     # RoFRC 对比模型（仅 FewRel 入口支持）
├── toolkit/
│   ├── __init__.py
│   ├── framework.py                 # 通用模型基类和训练/评估框架
│   ├── framework_fewrel.py          # FewRel checkpoint 指标策略
│   └── framework_tacred.py          # FS-TACRED checkpoint 指标策略
├── scripts/
│   ├── run_fewrel_baseline.sh       # FewRel 单组训练/测试配置脚本
│   └── run_tacred_baseline.sh       # FS-TACRED 多 shot/seed 基线实验脚本
└── data/
    ├── fewrel/
    │   ├── README.md
    │   └── description_pool.json    # FewRel 关系名称、原始描述、标准化描述
    └── fs_tacred/
        ├── README.md
        └── description_pool.json    # TACRED 关系描述池
```

训练后还会按需生成以下未预置目录：

- `checkpoint/`：保存 `*.pth.tar` 模型参数。
- `log/`：保存 FS-TACRED 基线日志、汇总 TSV 和边界校准结果。

## 3. 模块职责

### 3.1 训练入口

#### `train_fewrel.py`

FewRel 的总入口，职责包括：

1. 解析数据、episode、优化器和 BaFRC/RoFRC 超参数。
2. 固定 Python、NumPy、PyTorch 和 CUDA 随机种子，并启用确定性 cuDNN。
3. 创建 `BERTSentenceEncoder`。
4. 创建训练和验证的 `FewRelDataset` 迭代器。
5. 根据 `--model` 构建：
   - `BaFRC`（默认）；
   - `RoFRC`；
   - `MRM` 仅作为 `BaFRC` 的旧名称别名。
6. 执行训练或 `--only_test` 测试。
7. 普通测试分别在 `na_rate=1/2/5` 下评估，对应不同强度的 NOTA 查询。
8. `--test_online` 模式读取 FewRel 在线 episode，输出预测 JSON；模型内部的 NOTA 类 `N` 会转换为提交格式的 `-1`。

RoFRC 要求每个关系只提供一条描述，因此入口会自动把 `use_std_desc` 设为 `False`。

#### `train_tacred.py`

FS-TACRED 的总入口，只支持 `BaFRC`（`MRM` 同样是旧别名）。主要职责为：

1. 读取训练、验证和测试的预采样 episode 文件。
2. 强制 `na_rate=0`，因为 NOTA 标签已由 episode 内容决定，而不是运行时额外采样。
3. 训练时使用 `target_micro_f1` 选择最佳 checkpoint，并支持 early stopping。
4. 正常评估一份指定的测试 episode 文件。
5. 在 `--only_test --boundary_calibration` 模式下，一次编码每个 episode，然后对多个 `eval_margin` 复用距离缓存进行推理边界扫描，结果写入：

   ```text
   log/calibration/tacred/{N}way_{K}shot/seed{tag}.tsv
   ```

### 3.2 数据层 `dataset/`

#### `FewRelDataset.py`

- 从 `{root}/{name}.json` 读取按关系 ID 分组的 FewRel 样本。
- 每次 `__getitem__` 随机选择 N 个目标关系。
- 每类无放回采样 `K + Q` 条：K 条进入 support，Q 条进入 query。
- 额外从非目标关系中采样 `Q_na = int(na_rate * Q)` 条 NOTA query。
- 已知类标签为 `0 ... N-1`，NOTA 标签为 `N`。
- `__len__` 返回一个极大值，使 DataLoader 可持续生成随机 episode。
- `use_std_desc=True` 时每个已知类加入“原始描述 + 标准化描述”；关闭时只加入原始描述。

#### `FewRelOnlineDataset.py`

- 读取 CodaLab 风格的固定 episode：`meta_train`、`meta_test`、`relation`。
- `relation` 中的关系 ID 顺序必须与 `meta_train` 的类别顺序一致。
- 每个 episode 当前只使用一条 `meta_test` 查询。
- 始终为每个类构造原始/标准化两条描述，并追加两条 NOTA 描述，因此主要与 `use_std_desc=True` 的 BaFRC 形状约定匹配。

#### `FSTacredEpisodeDataset.py`

- 读取顶层形如 `[episodes, ...]` 的预采样 FS-TACRED JSON，实际使用 `raw[0]`。
- 每个 episode 包含形状为 `(N, K)` 的 `meta_train` 和查询列表 `meta_test`。
- 使用 support 样本中的 `relation` 字段确定当前 episode 的 N 个关系。
- query 的关系若在 support 关系列表中，则标签为对应类索引；否则为 NOTA（标签 `N`）。
- 如果缺少 `description_pool.json`，会用关系 ID 自动构造 TACRED 描述文本。
- DataLoader 外包裹无限迭代器，文件读到末尾后会从头继续，以满足固定迭代次数的训练/评估逻辑。

三种 collate 函数都会把多个 episode 展平为张量字典，主要字段是：

| 输入 | 字段 | 含义 |
|---|---|---|
| support/query | `word` | BERT token ID |
| support/query | `pos1` | 头实体标记位置 |
| support/query | `pos2` | 尾实体标记位置 |
| support/query | `mask` | attention mask |
| relation text | `word` | 关系名称与描述的 token ID |
| relation text | `mask` | 关系文本 attention mask |

### 3.3 编码层 `encoder/`

`BERTSentenceEncoder` 封装 Hugging Face `BertModel` 和 `BertTokenizer`。

句子编码流程：

- 在头实体边界插入 `[unused0]` / `[unused2]`。
- 在尾实体边界插入 `[unused1]` / `[unused3]`。
- 取两个实体起始标记位置的 BERT hidden state 并拼接，得到维度为 `2 × hidden_size` 的句子表示。
- 默认 `hidden_size=768`，因此句子表示通常为 1536 维。

关系文本编码格式为：

```text
[CLS] relation_name [SEP] relation_description
```

模型调用 `cat=False` 时分别取得 pooled/CLS 表示和逐 token hidden state，之后再拼接全局与平均局部表示。

### 3.4 模型层 `models/`

#### `BaFRC`

BaFRC 的核心由“描述增强原型 + 类别边界拒识”组成。

1. **构造类别原型**

   当使用标准化描述时：

   ```text
   p_c = (Σ support_c + relation_orig_c + relation_std_c) / (K + 2)
   ```

   消融标准化描述时：

   ```text
   p_c = (Σ support_c + relation_orig_c) / (K + 1)
   ```

2. **计算距离和已知类 logits**

   支持欧氏距离和余弦距离，默认欧氏距离：

   ```text
   logits_nway = -cls_temp × distance(query, prototype)
   ```

3. **估计类别半径**

   - 推理半径 `R_sup`：由其他类别 support 到当前原型的距离分位数减 margin 得到。
   - 训练半径 `R_train`：融合 support 半径和负 query 半径：

     ```text
     R_train = (1-rho) × R_sup + rho × R_qry
     ```

   - FewRel 基线使用 `rho=1.0`（纯 query 半径）。
   - FS-TACRED 基线使用 `rho=0.5`（support/query 混合半径）。

4. **开放集预测**

   先选距离最近的已知类。如果最小距离大于该类半径，则将结果改判为 NOTA（类别索引 `N`）。

5. **损失函数**

   ```text
   L = L_CE + L_B
   ```

   - `L_CE` 只在已知类 query 上计算 N-way 交叉熵。
   - `L_B` 同时惩罚落到边界外的正样本，以及进入边界内的负样本；正负两部分可独立关闭做消融。
   - `radius_reg`、`bafrc_loss_radius_reg` 以及若干旧参数目前只为命令兼容保留，不参与现行损失计算。

模型会在 `_cache` 中保存 query 距离、support 半径、support 表示和原型，供损失、联合拒识和边界校准复用。

#### `RoFRC`

RoFRC 是适配到当前框架接口的对比模型：

- support 原型为每类 K 个 support 表示的均值。
- 分别计算 query 与 support 原型、关系描述的点积。
- 已知类 logit 是两种点积的平均值。
- NOTA logit 只来自 NOTA 关系文本。
- 最终直接在 `N+1` 类上做 argmax 和交叉熵。
- 原论文中的 agreement loss 在当前适配代码中关闭，`lamb` 仅为接口兼容保留。

### 3.5 训练与评估层 `toolkit/`

`framework.py` 提供：

- `FewShotREModel`：持有句子编码器、默认交叉熵、准确率与辅助归一化方法。
- `FewShotREFramework`：统一训练、验证、离线评估和 FewRel 在线预测。

训练细节：

- 优化器：Transformers `AdamW`。
- 学习率：线性 warmup + 线性衰减。
- LayerNorm 和 bias 不施加 weight decay。
- 梯度裁剪阈值为 10。
- 支持梯度累积 `grad_iter`。
- checkpoint 仅保存 `{'state_dict': model.state_dict()}`，不保存 optimizer、scheduler 或当前迭代状态；`load_ckpt` 因而是权重初始化/继续微调，而非完整断点续训。
- 加载 checkpoint 时会忽略模型中不存在的键；普通评估未统一检查 shape，在线测试路径会额外跳过 shape 不一致的键。

两个薄封装决定最佳 checkpoint 的指标：

| 框架 | 选择指标 |
|---|---|
| `FewShotREFrameworkFewRel` | `accuracy + macro_f1`（legacy） |
| `FewShotREFrameworkTacred` | `target_micro_f1` |

评估输出包括：

- Accuracy；
- episode 平均 Macro-F1；
- 已知类样本准确率；
- 将 NOTA 视为正类的 Precision / Recall / F1；
- 将各已知目标关系合并统计的 Target Micro Precision / Recall / F1；
- gold known / predicted known 比例。

框架还保留了可选的 joint reject 逻辑，可联合“最近类边界余量”和“第一、第二近类距离差”重算 NOTA，但当前两个训练入口没有暴露对应命令行参数。

### 3.6 实验脚本 `scripts/`

#### `run_fewrel_baseline.sh`

- 默认执行 5-way 1-shot、seed 5、30,000 次训练迭代。
- 通过脚本顶部变量切换训练、仅测试或在线测试。
- 训练 NOTA 设置默认为 `NA_RATE=5`；训练完成后 Python 入口自动测试 `na_rate=1/2/5`。
- 可配置 checkpoint、encoder checkpoint 和日志文件。

#### `run_tacred_baseline.sh`

- 管理 1-shot/5-shot 与多个随机种子的实验矩阵。
- 默认脚本当前配置实际只运行 `SHOTS=5`、`SEEDS=5`；注释中给出了恢复 1/5-shot 和五个 seed 的方式。
- 每个模型 checkpoint 会在五份独立测试 episode seed 上评估。
- 从日志提取 ACC、Macro-F1、Target P/R/F1，写入：

  ```text
  log/baseline/tacred/summary.tsv
  ```

- 支持通过环境变量设置 `SEEDS`、`SHOTS`、`SKIP_EXISTING`、`EVAL_ONLY` 和 `TACRED_ROOT`。

## 4. 数据格式约定

### 4.1 关系描述池

两套 `description_pool.json` 都采用：

```json
{
  "relation_id": [
    "relation name",
    "original description",
    "standardized description"
  ]
}
```

第三项可缺失；关闭 `use_std_desc` 时不会使用它。

### 4.2 关系实例

编码器至少依赖以下字段：

```json
{
  "tokens": ["token_1", "token_2"],
  "h": ["head text", "head id", [[0]]],
  "t": ["tail text", "tail id", [[1]]]
}
```

其中 `h[2][0]` 与 `t[2][0]` 是头、尾实体的 token 位置列表。FS-TACRED 样本还使用 `relation` 字段判定 query 标签。

### 4.3 Episode 的主要张量形状

设 batch size 为 `B`，每个 episode 有 N 类、每类 K 个 support、实际 query 数为 `T`，编码维度 `d = 2 × hidden_size`：

| 张量 | 典型形状 |
|---|---|
| 展平后的 support 输入 | `(B×N×K, max_length)` |
| 展平后的 query 输入 | `(B×T, max_length)` |
| support 表示 | `(B, N, K, d)` |
| query 表示 | `(B, T, d)` |
| 类别原型 | `(B, N, d)` |
| query-to-prototype 距离 | `(B, T, N)` |
| BaFRC 类别半径 | `(B, N)` |
| BaFRC 预测 | `(B, T)`，值域 `0...N` |

关系文本数量取决于数据路径和 `use_std_desc`：已知类通常为 `N` 或 `2N` 条；FewRel 含 NOTA 描述时为 `N+1` 或 `2N+2` 条。BaFRC 只取前 N 类对应的描述构造原型，NOTA 本身通过半径拒识产生。

## 5. 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

准备兼容的 BERT checkpoint 与数据后，在仓库根目录运行：

```bash
bash scripts/run_fewrel_baseline.sh
bash scripts/run_tacred_baseline.sh
```

也可直接调用入口，例如：

```bash
python train_fewrel.py --root ./data/fewrel --N 5 --K 1 --Q 1
python train_tacred.py --root ./data/fs_tacred --N 5 --K 5 --Q 1
```

实际运行前必须补齐脚本所引用的 JSON 数据文件。当前仓库的数据目录中仅有说明文件与描述池。

## 6. 依赖与运行环境

`requirements.txt` 声明：

- `torch`
- `transformers`
- `numpy`
- `scikit-learn`
- `tqdm`

代码自动检测 CUDA：有可用 GPU 时会将模型及 batch 张量移动到 GPU，否则使用 CPU。基线规模和 BERT 编码成本较高，实际实验更适合 CUDA 环境。

## 7. 当前工程特征与注意事项

- 仓库没有自动化测试、配置文件体系或 Python 包构建文件，实验参数主要位于命令行和 shell 脚本中。
- `dataset/`、`encoder/`、`models/`、`toolkit/` 下的 `__init__.py` 主要用于包识别，核心实现均在同目录的具体模块中。
- `Q` 对 FS-TACRED 更多是配置提示；框架会根据 batch 中实际 `query_label` 数量反推每个 episode 的 query 数。
- BaFRC 推理只使用 support 校准半径，而训练边界损失可混合 query 信息；`radius_blend_rho` 决定两者差异程度。
- `radius_detach` 在 FewRel 入口可配置，默认阻断半径估计过程的梯度；FS-TACRED 入口未暴露该参数，使用模型默认值。
- 部分构造参数（如 `use_query_in_radius`、`use_desc_neg`、`use_query_in_boundary`、`radius_update_interval`）为旧实验接口兼容项，当前 `BaFRC` 主路径并未读取它们。
- 代码依赖 `transformers.AdamW`；若使用较新的 Transformers 版本，应确认该导入路径仍兼容，仓库当前没有锁定依赖版本。
- shell 脚本面向 Bash；在 Windows PowerShell 下通常需要 Git Bash、WSL 或手动转换为 Python 命令。

## 8. 快速定位指南

| 想修改的内容 | 首选文件 |
|---|---|
| BaFRC 原型、半径、拒识和边界损失 | `models/bafrc.py` |
| RoFRC 对比逻辑 | `models/RoFRC.py` |
| 实体标记与 BERT 表示 | `encoder/BertEncoder.py` |
| FewRel episode / NOTA 采样 | `dataset/FewRelDataset.py` |
| FewRel 在线提交输入 | `dataset/FewRelOnlineDataset.py` |
| FS-TACRED episode 与标签映射 | `dataset/FSTacredEpisodeDataset.py` |
| 优化、验证、指标、checkpoint | `toolkit/framework.py` |
| FewRel 命令行参数与三档 NOTA 测试 | `train_fewrel.py` |
| FS-TACRED 参数与边界校准 | `train_tacred.py` |
| 复现实验组合与默认超参数 | `scripts/*.sh` |

