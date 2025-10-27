# SFT 项目规划

## 项目目标
- 构建一个可在指定模型路径上进行指令微调（SFT）的工程化项目，支持 LoRA 与全量微调两种模式，并可通过超参数一键切换。
- 面向 MATH 与 GSM8K 数据集，提供系统化的数据准备、训练、断点续训与评估能力。
- 在训练过程中按既定策略保存检查点（ckpt），并支持加载 ckpt 进行离线评估，输出可复现的准确率结果。

## 约束与假设
- 依赖 Hugging Face Transformers、PEFT、Accelerate/DeepSpeed 或 FSDP 等主流工具链；与 LESS 现有脚本保持兼容，复用 `less/scripts/train/base_training_args.sh` 中的公共参数。
- 默认使用单机多卡环境，必要时支持 FSDP/DeepSpeed；确保可在无外网环境下运行（需提前下载模型与数据）。
- 数据集统一转为标准化 JSONL/Parquet，字段包含 `instruction`、`input`、`output` 或 `question`、`answer`。

## 项目目录规划
```
SFT_project/
├── configs/               # 全量/LoRA/评估等 YAML 配置
├── data/
│   ├── raw/               # 原始数据（MATH、GSM8K）
│   ├── processed/         # 统一格式后的训练/验证/测试集
│   └── prompts/           # 评估提示词模板
├── docs/
│   └── reports/           # 评估报告、实验记录
├── logs/
├── outputs/
│   ├── checkpoints/       # 训练 ckpt，按模型/时间戳组织
│   └── eval/              # 评估结果与预测文件
├── scripts/
│   ├── train/             # 训练入口：train.py、lora_train.sh、full_train.sh
│   └── evaluate/          # 评估入口与结果解析
├── src/
│   ├── data/              # 数据处理、数据集封装
│   ├── models/            # 模型包装、LoRA 注入
│   ├── training/          # Trainer、加速器、回调
│   └── evaluation/        # 指标计算、答案解析
└── README.md
```

## 数据准备流程
- **统一抽象**：在 `src/data/` 实现通用数据加载与预处理模块，支持传入数据配置自动完成读取、清洗、字段映射、tokenizer 预处理缓存。
- **数据切分**：按 8:1:1 或自定义比例划分 train/valid/test，并记录随机种子以保持可复现性。
- **MATH 数据集**：
  - 下载/同步官方 JSON 文件。
  - 解析题干、标准答案，对答案进行 LaTeX->纯文本转换与规范化。
  - 可选：构造基于难度的分层采样，缓解训练偏差。
- **GSM8K 数据集**：
  - 读取 `train.jsonl` 与 `test.jsonl`，转换为统一 schema。
  - 对解析步骤与最终答案分离存储，以支持链路输出评估。
- **特征缓存**：针对大模型训练，在首次 tokenize 后缓存到磁盘（Arrow/pt 文件），避免重复处理。

## 配置管理
- 在 `configs/` 下维护多份 YAML：
  - `model.yaml`：基础模型路径、tokenizer、padding 设置。
  - `train_lora.yaml` 与 `train_full.yaml`：学习率、batch size、LoRA r/alpha/target modules、梯度累积、FSDP/DeepSpeed 参数。
  - `data_math.yaml`、`data_gsm8k.yaml`：数据路径、处理策略、评估模板。
  - `eval.yaml`：评估批大小、最大生成长度、判分规则。
- 主训练脚本读取通用配置 + 模式配置，通过命令行参数 `--use_lora true|false`、`--datasets math gsm8k` 等组合。

## 训练流水线设计
- **入口脚本**：`scripts/train/train.py`
  - 解析命令行参数（模型路径、use_lora、数据集列表、输出目录、ckpt 恢复等）。
  - 根据 `use_lora` 动态加载 LoRA 或全量配置，实例化相应 Trainer。
  - 与 LESS 中的 `header`（环境变量/accelerate 启动串）兼容，可被 shell 脚本包装调度。
- **共用组件**：
  - `src/models/base_model.py`：封装模型加载、权重保存逻辑。
  - `src/training/callbacks.py`：自定义保存回调（最优/间隔）、学习率监控、early stopping。
  - `src/training/accelerator.py`：统一处理 Accelerator/FSDP/DeepSpeed 初始化。
- **LoRA 分支**：
  - 使用 PEFT 注入 LoRA（target modules 随模型自动匹配）。
  - 仅保存 LoRA 权重与合并权重两种模式（配置化）。
  - 训练结束后可选择 `merge_and_unload()` 导出全量权重用于推理。
- **全量分支**：
  - 加载基础模型后全参数参与训练，可结合 FSDP/DeepSpeed 进行显存优化。
  - 提供梯度检查点、混合精度、梯度裁剪选项。
- **监控与日志**：
  - 集成 WandB 或 TensorBoard（可选开关）；默认记录至 `logs/`。
  - CLI 输出同时写入 `outputs/checkpoints/<job_name>/train.log`，与现有 LESS 脚本保持一致。

## Checkpoint 策略
- 统一使用 `TrainerCallback` 或 Accelerate Hook 实现：
  - 定期（如每 N global_step）保存滚动 ckpt。
  - 根据验证集指标（accuracy / loss）保存 `best` ckpt。
  - 保存最新 ckpt 的同时记录训练状态（optimizer、scheduler、随机种子）。
- 提供 `--resume_from_checkpoint` 参数支持断点续训；自动检测 ckpt 完整性。
- 输出结构示例：
```
outputs/checkpoints/
└── <job_name>/
    ├── config.json
    ├── checkpoint-000100/
    ├── checkpoint-best/
    └── train.log
```

## 评估流程
- **整体策略**：加载目标 ckpt，针对 MATH 与 GSM8K 生成答案，使用标准化判分模块计算准确率与题目级别结果。
- **推理脚本**：`scripts/evaluate/run_eval.py`
  - 参数：模型路径/ckpt、数据集名称、batch size、max_new_tokens、temperature。
  - 可选择是否开启链式思维（CoT）提示模板。
- **答案解析与判分**：
  - MATH：
    - 生成结果经正则去除多余文本后提取最终答案（支持 `\boxed{}`、分数、小数）。
    - 与标准答案比对，容忍浮点误差或格式差异。
  - GSM8K：
    - 使用常见的 `Answer:` 行抽取数值，或基于自然语言解析。
    - 对字符串化数字做归一化（移除逗号、空格、单位）。
  - 记录每题预测、是否正确、错误类型，输出到 `outputs/eval/<dataset>/<run_id>.json`。
- **指标输出**：
  - 汇总总体准确率、子类别准确率、平均解题长度。
  - 自动生成 Markdown/CSV 报告，存放于 `docs/reports/`。

## 自动化与测试
- 为数据处理、答案解析等关键模块编写单元测试，确保格式变更不会破坏流程。
- 使用 `Makefile` 或 `invoke` 统一封装 `make preprocess`, `make train`, `make eval` 等命令。
- 可选加入 CI（如 GitHub Actions）执行快速单测与 lint。

## 里程碑与交付
1. **环境与骨架**：完成目录初始化、依赖声明、基础 README，确保训练脚本可加载模型并跑通干运行。
2. **数据准备**：实现 MATH/GSM8K 预处理与缓存，生成 sample batch 验证。
3. **训练实现**：落地 LoRA 与全量训练流程，完成小规模 smoke test，确认 ckpt 写入正确。
4. **评估模块**：实现推理、答案解析与指标统计，使用基线模型验证准确率计算。
5. **优化与文档**：补充配置模板、最佳实践、实验记录表，整理最终交付材料。

## 风险与应对
- **显存瓶颈**：提前验证 LoRA 与全量配置的显存占用，准备梯度累积与混合精度方案。
- **数据质量**：针对标注错误或格式不一致，建立验证脚本；必要时人工抽查。
- **评估鲁棒性**：对解析失败的样本进行错误日志记录，并支持人工复查与重评分。
- **训练不稳定**：监控 loss/梯度爆炸，提供自动学习率调节与梯度裁剪。

## 后续扩展建议
- 集成多任务训练（MATH+GSM8K 混合/交替采样）。
- 支持 QLoRA、LoRA Rank 逐层搜索、或 Adapter 混合策略。
- 加入自动化超参搜索与实验追踪（Ray Tune/Optuna）。
