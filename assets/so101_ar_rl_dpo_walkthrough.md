# SO101 AR-RL / Window DPO 训练机制讲义

这份笔记解释当前仓库里的 `scripts/experiments/so101_ar_rl` 实验，以及它和
`D:\workspace\GRAPE` 里的 TPO/DPO 代码是什么关系。结论先放前面：

- 这里的“RL”不是在线 RL。训练时模型不再和机器人交互，而是在离线采好的
  success/failure 轨迹上做 preference optimization。
- 当前主实验是 `ar_dpo`：把同任务、同初始化 bucket 内的成功轨迹当作
  chosen，把失败轨迹当作 rejected，展开成固定长度窗口，然后对窗口内的
  G0.5 自回归 action-token logprob 做 DPO。
- `TPO` 在 GRAPE 代码里本质上是 trajectory-wise preference optimization：
  换句话说，loss 仍是 DPO 形式，只是比较对象不是一条语言 response，而是一段
  轨迹窗口。
- 当前 G0.5 实现没有直接搬 GRAPE 的 OpenVLA/RLDS/LoRA 训练栈，而是在现有
  LeRobot + G05Policy 训练栈中重做了同一个思想。

## 1. 端到端训练链路

最终会用到的脚本是：

1. `scripts/experiments/so101_ar_rl/train_ar_dpo.sh`
   - 前台 DPO 训练入口。
   - 会自动跑 data prepare 和 pair build。
   - 会自动确保 reference logp cache 存在。
   - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:46-55`,
     `scripts/experiments/so101_ar_rl/train_ar_dpo.sh:81-96`。

2. `scripts/experiments/so101_ar_rl/cache_ref_logps.sh`
   - 只生成 reference logp cache，不训练。
   - 代码位置：`scripts/experiments/so101_ar_rl/cache_ref_logps.sh:37-42`,
     `scripts/experiments/so101_ar_rl/cache_ref_logps.sh:68-83`。

3. `scripts/experiments/so101_ar_rl/train_ar_sft_success.sh`
   - success-only SFT baseline。
   - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:37-45`,
     `scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:72-76`。

4. `scripts/experiments/so101_ar_rl/prepare_data.sh`
   - 把 raw LeRobot root 转成 G0.5 model frame。
   - 由 DPO/SFT 脚本自动调用。
   - 代码位置：`scripts/experiments/so101_ar_rl/prepare_data.sh:11-13`。

5. `scripts/experiments/so101_ar_rl/build_pairs.sh`
   - 根据 label 构造 DPO pair。
   - 由 DPO 脚本自动调用。
   - 代码位置：`scripts/experiments/so101_ar_rl/build_pairs.sh:14-18`。

后台版本 `run_all_background.sh` 只是 supervisor：它现在只启动 DPO，不再默认跑
SFT 对照，见 `scripts/experiments/so101_ar_rl/run_all_background.sh:23-31`。

## 2. 数据先从 raw frame 变成 G0.5 model frame

原始数据在：

```text
data/g05_rl_raw/so101_g05_rl_pick_white_v1
```

训练用 prepared 数据在：

```text
data/g05_rl_prepared/so101_g05_rl_pick_white_v1
```

`prepare_g05_rl_data.py` 做三件事：

- 找到每个 raw run root：`scripts/experiments/so101_ar_rl/tools/prepare_g05_rl_data.py:26-27`。
- 复制 `meta/`，软链接 `videos/` 和 `recording_context/`，不重编码视频：
  `scripts/experiments/so101_ar_rl/tools/prepare_g05_rl_data.py:75-78`。
- 改写 parquet 里的 `action` 和 `observation.state`，并改写 label 里的
  `dataset_dir` 到 prepared root：
  `scripts/experiments/so101_ar_rl/tools/prepare_g05_rl_data.py:35-54`,
  `scripts/experiments/so101_ar_rl/tools/prepare_g05_rl_data.py:79-88`。

坐标变换来自 SO101 SFT 准备脚本：

```python
SIGNS = [1, -1, 1, 1, 1, 1]
OFFSETS = [0, 90, 90, 0, 0, 0]
```

代码位置：`scripts/so101_square_finetune/prepare_g05_model_frame.py:32-33`。
真正的变换函数是：

\[
q_{\mathrm{model},d}
=
s_d q_{\mathrm{raw},d} + b_d,
\qquad d=1,\ldots,6
\]

其中：

- \(q_{\mathrm{raw},d}\)：raw SO101 第 \(d\) 个关节值。
- \(q_{\mathrm{model},d}\)：G0.5 训练/部署期望的第 \(d\) 个关节值。
- \(s=[1,-1,1,1,1,1]\)。
- \(b=[0,90,90,0,0,0]\)，单位是 degree。
- 代码实现：`model_frame(values) = values * SIGNS + OFFSETS`，
  见 `scripts/so101_square_finetune/prepare_g05_model_frame.py:57-60`。

统计文件也要变换。均值直接变换，标准差乘 \(|s_d|\)，min/max 和 quantile 在符号为
负时要交换两端，否则 q01/q99 会倒过来。代码位置：
`scripts/so101_square_finetune/prepare_g05_model_frame.py:63-87`。

当前 prepared summary 显示 5 个 run、7988 帧、53 个 label：
`data/g05_rl_prepared/so101_g05_rl_pick_white_v1/prepare_summary.json`。

## 3. Label 是 RL 数据的源头

本实验所有 success/failure、bucket、episode 过滤都从 label 文件来，而不是从
文件夹结构猜。旧上下文也记录过同一原则：`rl_rollout_labels.jsonl` 是 source of
truth。

加载 label 的函数会扫描：

- `rl_rollout_labels.jsonl`
- `rl_rollout_labels_prepared.jsonl`

同一个 parent 下 prepared 版本会覆盖普通版本：
`src/g05/rl/so101_preference_data.py:67-75`。

默认要过滤坏 episode：

```text
20260701_114422_ep00003
```

相关代码位置：

- 常量：`src/g05/rl/so101_preference_data.py:35`。
- 训练配置默认过滤：`scripts/finetune.py:414-421`。
- DPO/SFT 脚本显式传入：
  `scripts/experiments/so101_ar_rl/train_ar_dpo.sh:96`,
  `scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:76`。

当前 pair summary 显示：

- raw labels: 53
- filtered labels: 52
- success episodes: 27
- failure episodes: 25
- base pairs: 85

验证脚本中也写死了这些期望：
`scripts/experiments/so101_ar_rl/tools/check_data.py:41-63`。

## 4. Pair 构建：同 instruction、同 bucket 内 success × failure

DPO 的 pair 文件由：

```bash
bash scripts/experiments/so101_ar_rl/build_pairs.sh
```

生成。脚本调用：

```bash
python tools/build_rl_pairs.py \
  --labels-root data/g05_rl_prepared/so101_g05_rl_pick_white_v1 \
  --output cache/pairs.jsonl \
  --exclude-episode-uid 20260701_114422_ep00003 \
  --expect-pairs 85
```

代码位置：`scripts/experiments/so101_ar_rl/build_pairs.sh:14-18`。

构造规则在 `tools/build_rl_pairs.py`：

- 每条 label 转成 episode ref：
  `scripts/experiments/so101_ar_rl/tools/build_rl_pairs.py:41-49`。
- bucket key 是 `(instruction, init_config_id)`：
  `scripts/experiments/so101_ar_rl/tools/build_rl_pairs.py:76-85`。
- 每个 bucket 内做全交叉：
  `for chosen in success: for rejected in failure`：
  `scripts/experiments/so101_ar_rl/tools/build_rl_pairs.py:90-105`。

数学上，设某个 bucket \(b\) 中：

\[
S_b = \{e_i : e_i \text{ is success}\},
\qquad
F_b = \{e_j : e_j \text{ is failure}\}.
\]

这个 bucket 生成的 pair 集合是：

\[
\mathcal P_b
=
\{(e_i^+, e_j^-): e_i^+\in S_b,\ e_j^-\in F_b\}.
\]

总 pair 集合：

\[
\mathcal P
=
\bigcup_b \mathcal P_b,
\qquad
|\mathcal P|
=
\sum_b |S_b|\,|F_b|.
\]

这里 \(e_i^+\) 是 chosen episode，\(e_j^-\) 是 rejected episode。

注意：这和 GRAPE README 里的“一一配对”不同。GRAPE 要求 chosen dataset 的第
\(n\) 条轨迹对应 rejected dataset 的第 \(n\) 条轨迹，并且同任务、同初始状态：
`D:\workspace\GRAPE\README.md:99-102`。当前 G0.5 repo 则是在同 bucket 内全交叉，
所以一个 success 会和同 bucket 的多个 failure 组成多个 pair。

## 5. Pair 如何映射回 G0.5 dataset index

G0.5 的普通训练数据集是 `MixtureLerobotDataset`。它把多个 dataset group 拼成一个
全局 index 空间：

- 加载各 group：`src/g05/data/mixture_lerobot_dataset.py:161-213`。
- 记录每个 group 的有效长度和起点：
  `src/g05/data/mixture_lerobot_dataset.py:236-241`。
- `__getitem__` 用全局 index 解析到 `(dataset_idx, local_idx)`：
  `src/g05/data/mixture_lerobot_dataset.py:408-444`。

RL wrapper 做的是反向映射：从 label 的 `(dataset_dir, episode_index)` 找到
Mixture dataset 中对应的全局 frame range。

关键代码：

- 要求 base dataset 是 `MixtureLerobotDataset`：
  `src/g05/rl/so101_preference_data.py:139-144`。
- 要求 `data.use_weight_for_sampling=false`，否则 label episode index 和训练采样 index
  不再一一对应：
  `src/g05/rl/so101_preference_data.py:145-149`。
- 建 root offset：`src/g05/rl/so101_preference_data.py:165-175`。
- 用 `episode_data_index["from"]` / `["to"]` 得到 episode 局部范围：
  `src/g05/rl/so101_preference_data.py:189-201`。
- 如果 label 有 `frame_count`，还会把结尾 clamp 到 label 记录的 frame 数：
  `src/g05/rl/so101_preference_data.py:196-199`。
- 空 range 会跳过：
  `src/g05/rl/so101_preference_data.py:202-214`。

因此每个 episode 变成：

\[
e
\mapsto
(d,\ell_{\min},\ell_{\max}),
\]

其中：

- \(d\)：Mixture 中的 dataset group index。
- \(\ell_{\min}\)：这个 episode 在该 group 内的起始 local index。
- \(\ell_{\max}\)：结束 local index，不含右端点。
- episode 长度 \(T = \ell_{\max}-\ell_{\min}\)。

当前 default cache metadata 显示：

- 原始 pair 文件有 85 条。
- 实际可训练 base pair 是 78 条。
- 默认 `window_size=8, stride=4, relative, max=0` 展开后是 2409 个训练样本。

这个 2409 来自：
`scripts/experiments/so101_ar_rl/cache/ref_logps_step9740_window8_stride4_relative_max0.jsonl.meta.json`。

## 6. SFT wrapper：SuccessFrameDataset

`ar_sft_success` 不构造 chosen/rejected pair，而是只暴露成功 episode 的所有 frame。

代码位置：

- 入口选择：`scripts/finetune.py:434-439`。
- wrapper 实现：`src/g05/rl/so101_preference_data.py:258-303`。

设成功 episode 集合为：

\[
\mathcal E^+
=
\{e: \mathrm{success}(e)=1\}.
\]

SFT 数据集暴露的 frame 集合是：

\[
\mathcal D_{\mathrm{SFT}}
=
\{(d,\ell): e\in\mathcal E^+,\ \ell_{\min}(e)\le \ell<\ell_{\max}(e)\}.
\]

训练时 `train_ar_sft_success.sh` 会把 `+rl.mode=ar_sft_success` 传给
`scripts/finetune.py`，见
`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:72-76`。

一个容易误会的点：`SFT_ACTION_ONLY=false` 是脚本默认值，
`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:34-35`。此时训练走普通
`forward_train`，见 `scripts/finetune.py:1400-1410`。只有设置
`SFT_ACTION_ONLY=true` 时，才走 `forward_ar_sft_action_only`，显式只用 action-token
logprob，见 `src/g05/models/g05/g05_policy.py:807-823`。

不过该 SFT 脚本也显式关闭 continuous action：
`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:60-62`。DPO 脚本也同样关闭
continuous/return continuous：
`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:69-71`。

## 7. DPO wrapper：PreferencePairDataset

DPO 数据集从 `pairs.jsonl` 读 base pairs：

```python
self.raw_pairs = read_jsonl(self.pairs_path)
```

代码位置：`src/g05/rl/so101_preference_data.py:443-445`。

每个 base pair 可以用两种采样方式展开：

1. `sample_mode=anchor`
2. `sample_mode=window`

入口在 `scripts/finetune.py:440-456`。

### 7.1 Anchor mode

`anchor` 模式用于旧的稀疏帧方案。函数：
`src/g05/rl/so101_preference_data.py:310-334`。

如果 episode 长度 \(T\) 大于 anchor 数 \(K\)，它取：

\[
r_k
=
\mathrm{round}\left(
\alpha_k (T-1)
\right),
\qquad
\alpha_k\in\mathrm{linspace}(0.15,0.85,K).
\]

然后把相对 offset \(r_k\) 变成 local index：

\[
\ell_k
=
\ell_{\min}+r_k.
\]

代码还会去重，如果短 episode 里 round 后冲突，就从开头补齐。

### 7.2 Window mode

当前主实验默认是 `window`：

```bash
DPO_SAMPLE_MODE=window
WINDOW_SIZE=8
WINDOW_STRIDE=4
WINDOW_ALIGN=relative
MAX_WINDOWS_PER_PAIR=0
```

代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:12-18`。

窗口生成函数是 `_window_start_pairs`：
`src/g05/rl/so101_preference_data.py:351-413`。

对一个 base pair：

- chosen episode 长度 \(T^+\)。
- rejected episode 长度 \(T^-\)。
- 配置窗口长度 \(w\)。
- 实际窗口长度：

\[
\tilde w
=
\max\left(1,\min(w,T^+,T^-)\right).
\]

代码位置：`src/g05/rl/so101_preference_data.py:372-375`。

每条轨迹可放窗口的起点数：

\[
P^+ = \max(1,T^+-\tilde w+1),
\qquad
P^- = \max(1,T^--\tilde w+1).
\]

代码位置：`src/g05/rl/so101_preference_data.py:376-377`。

#### index 对齐

如果 `window_align=index`，代码取：

\[
P = \min(P^+,P^-).
\]

然后起点是：

\[
0,\ s,\ 2s,\ldots
\]

其中 \(s=\texttt{WINDOW\_STRIDE}\)。如果最后一个起点不是 \(P-1\)，代码会追加
最后窗口，确保尾部被覆盖：
`src/g05/rl/so101_preference_data.py:381-389`。

这最像 GRAPE 的直接 sliding window：第 \(i\) 个 chosen window 配第 \(i\) 个
rejected window。

#### relative 对齐

默认是 `window_align=relative`。它先算两边各自 stride 后大约有多少窗口：

\[
C^+ = \left\lceil \frac{P^+}{s} \right\rceil,
\qquad
C^- = \left\lceil \frac{P^-}{s} \right\rceil,
\qquad
C = \max(1,\min(C^+,C^-)).
\]

代码位置：`src/g05/rl/so101_preference_data.py:390-394`。

然后取 \(C\) 个相对进度：

\[
\rho_c \in \mathrm{linspace}(0,1,C).
\]

若 \(C=1\)，代码用 \(\rho=0.5\) 取中间位置。

chosen/rejected 起点分别是：

\[
t_c^+
=
\mathrm{round}\left(\rho_c(P^+-1)\right),
\qquad
t_c^-
=
\mathrm{round}\left(\rho_c(P^--1)\right).
\]

代码位置：`src/g05/rl/so101_preference_data.py:394-403`。

这解决了成功轨迹和失败轨迹长度不同的问题：它不是死配相同 index，而是配相同相对进度。

### 7.3 每个 window 变成训练 item

对一个 base pair \(p\) 的第 \(c\) 个窗口，训练 item 是：

\[
\left(
y^+_{t_c^+:t_c^++\tilde w-1},
y^-_{t_c^-:t_c^-+\tilde w-1}
\right).
\]

其中：

- \(y^+\)：chosen 轨迹中的 action-token target 序列。
- \(y^-\)：rejected 轨迹中的 action-token target 序列。
- 每个窗口 item 的 `pair_id` 会变成 `base_pair_id_w0000` 这样的格式。

实现位置：
`src/g05/rl/so101_preference_data.py:476-506`。

`__getitem__` 会把窗口里的每个 frame 都读成普通 G0.5 sample：
`src/g05/rl/so101_preference_data.py:558-593`。

## 8. Collate：window 内 frame 被 flatten，靠 anchor_counts 还原

`collate_preference_pairs` 做了一个关键操作：把 batch 中所有 chosen window 的 frame
摊平成一个普通 collated batch，把所有 rejected window 也摊平成另一个 collated batch。

代码位置：
`src/g05/rl/so101_preference_data.py:600-630`。

假设 DataLoader batch size 是 \(B\)，第 \(j\) 个 pair/window 里有 \(K_j\) 个 frame。
当前默认 `model.batch_size=1`，所以通常 \(B=1, K_1=8\)。

collate 后：

- `chosen_samples` 长度是 \(\sum_j K_j\)。
- `rejected_samples` 长度是 \(\sum_j K_j\)。
- `anchor_counts = [K_1,\ldots,K_B]`。
- `ref_chosen_logp` / `ref_rejected_logp` 是每个 window pair 的 reference score。

后面模型先得到每个 frame 的 logprob：

\[
\ell_{j,r}^{+}
=
\log \pi_\theta(y_{j,r}^{+}\mid x_{j,r}^{+}),
\qquad
r=1,\ldots,K_j.
\]

再用 `aggregate_anchor_logps` 按 `anchor_counts` 加回每个 pair/window：

\[
S_\theta^+(j)
=
\sum_{r=1}^{K_j}\ell_{j,r}^{+},
\qquad
S_\theta^-(j)
=
\sum_{r=1}^{K_j}\ell_{j,r}^{-}.
\]

代码位置：
`src/g05/rl/so101_preference_data.py:633-635`,
`src/g05/models/g05/g05_policy.py:757-760`。

这里的 \(S_\theta^+\) 和 \(S_\theta^-\) 就是 DPO 里用的 policy logprob。

## 9. Reference logp cache

标准 DPO 需要 reference model：

\[
\pi_{\mathrm{ref}}.
\]

GRAPE 训练时同时加载 policy model 和 reference model：
`D:\workspace\GRAPE\TPO-Train\finetune.py:592-605`。

G0.5 模型显存更重，所以当前仓库不同时加载两份模型。它先用 base checkpoint 算好每个
pair/window 的 reference logprob，写到 JSONL cache。

入口在 `scripts/finetune.py`：

- 构造 cache metadata：
  `scripts/finetune.py:460-476`。
- 训练前确保 cache：
  `scripts/finetune.py:932-957`。

真正计算在：
`src/g05/rl/so101_preference_data.py:638-708`。

每行 cache 是：

```json
{
  "pair_id": "...",
  "ref_chosen_logp": ...,
  "ref_rejected_logp": ...,
  "anchor_count": ...
}
```

数学上：

\[
S_{\mathrm{ref}}^+(j)
=
\sum_{r=1}^{K_j}
\log \pi_{\mathrm{ref}}(y_{j,r}^{+}\mid x_{j,r}^{+}),
\]

\[
S_{\mathrm{ref}}^-(j)
=
\sum_{r=1}^{K_j}
\log \pi_{\mathrm{ref}}(y_{j,r}^{-}\mid x_{j,r}^{-}).
\]

cache metadata 会包含：

- base checkpoint path
- pair file path
- pair file sha256
- sample mode
- window size/stride/alignment
- train sample count
- excluded episode uid

代码位置：`scripts/finetune.py:460-476`。
如果 metadata 不匹配，cache 会重算：
`src/g05/rl/so101_preference_data.py:654-663`。

当前默认 cache：

```text
scripts/experiments/so101_ar_rl/cache/ref_logps_step9740_window8_stride4_relative_max0.jsonl
```

有 2409 行，对应 2409 个 window training samples。

## 10. G0.5 如何计算 action-token logprob

DPO 只比较 G0.5 自回归 action-token likelihood。核心函数：

```python
G05Policy.compute_ar_action_logps
```

代码位置：`src/g05/models/g05/g05_policy.py:655-692`。

过程：

1. `processor.encode_train(...)` 把 sample 转成：
   - `input_ids`
   - `labels`
   - `attention_mask`
   - `split_index`

   代码位置：`src/g05/models/g05/g05_policy.py:666-672`。

2. `process_pixel_values` 处理图像。

3. 如果启用 proprio encoder，构造 proprio batch：
   `src/g05/models/g05/g05_policy.py:674-683`。

4. `self.model.vlm_prefill(...)` 得到 VLM hidden states：
   `src/g05/models/g05/g05_policy.py:684-691`。

5. `ar_helper.cal_action_logps(...)` 从 hidden states 和 labels 计算 action-token logprob：
   `src/g05/models/g05/g05_policy.py:692`。

`encode_train` 的输入序列由 prefix + suffix 拼成：
`src/g05/models/g05/io/input_preprocessor.py:950-968`。

各段文本/图像/action 怎么设 label：

- 静态文本如果 `loss_on_static_text=false`，label 是 `IGNORE_INDEX=-100`：
  `src/g05/models/g05/io/input_preprocessor.py:551-558`。
- action 段由 `BuiltinActionProcessor` 调用 BAR/action tokenizer 生成 token 和 label：
  `src/g05/models/g05/io/input_preprocessor.py:1572-1595`。

`cal_action_logps` 再做更严格的过滤：

- language model 是 next-token prediction，所以先 shift：
  `shift_hidden = hidden[:, :-1]`，`shift_labels = labels[:, 1:]`。
- 只保留 `labels != -100` 的位置。
- 如果配置了 action-token id range，再只保留 action token。

代码位置：
`src/g05/models/g05/helpers/ar_helper.py:301-358`。

数学上，对一个 sample \(i\)：

- 序列长度是 \(L_i\)。
- target label 序列是 \(z_{i,1:L_i}\)。
- \(-100\) 表示忽略该 token。
- action-token 有一组 token id 范围，代码用 `action_token_range` 过滤：
  `src/g05/models/g05/helpers/ar_helper.py:318-321`。

定义有效 action-token 位置集合：

\[
\mathcal A_i
=
\{k:
z_{i,k}\ne -100
\ \text{and}\ 
z_{i,k}\in\mathrm{ActionTokenRange}\}.
\]

由于 next-token prediction，位置 \(k\) 的 label 由位置 \(k-1\) 的 hidden state 预测。
设：

- \(h_{i,k-1}\)：VLM 在位置 \(k-1\) 的 hidden state。
- \(g_\theta(h_{i,k-1})\in\mathbb R^{V}\)：decode 到 vocab logits。
- \(V\)：词表大小。
- \(z_{i,k}\)：第 \(k\) 个目标 token id。

该 token 的 logprob 是：

\[
\log \pi_\theta(z_{i,k}\mid x_i,z_{i,<k})
=
\log
\frac{
\exp(g_\theta(h_{i,k-1})_{z_{i,k}})
}{
\sum_{v=1}^{V}\exp(g_\theta(h_{i,k-1})_v)
}.
\]

sample 级 action logprob 是：

\[
\ell_i(\theta)
=
\sum_{k\in\mathcal A_i}
\log \pi_\theta(z_{i,k}\mid x_i,z_{i,<k}).
\]

代码里通过 cross entropy 得到每个 token 的 negative log likelihood：

\[
\mathrm{CE}_{i,k}
=
-\log \pi_\theta(z_{i,k}\mid x_i,z_{i,<k}),
\]

然后 scatter 到 sample owner 上：

\[
\ell_i(\theta)
=
\sum_{k\in\mathcal A_i} -\mathrm{CE}_{i,k}.
\]

代码位置：
`src/g05/models/g05/helpers/ar_helper.py:346-357`。

返回值：

- `logps`: 每个 sample 的 \(\ell_i(\theta)\)。
- `token_counts`: 每个 sample 的 \(|\mathcal A_i|\)。
- `ce_loss`: 有效 action token 的平均 NLL。
- `mean_token_logp`: \(\ell_i/|\mathcal A_i|\)。

## 11. DPO 目标函数

对一个训练样本 \(j\)，也就是一个 chosen/rejected window pair：

- chosen score:

\[
S_\theta^+(j)
=
\sum_{r=1}^{K_j}
\ell_{j,r}^{+}(\theta)
\]

- rejected score:

\[
S_\theta^-(j)
=
\sum_{r=1}^{K_j}
\ell_{j,r}^{-}(\theta)
\]

- reference chosen score:

\[
S_{\mathrm{ref}}^+(j)
=
\sum_{r=1}^{K_j}
\ell_{j,r}^{+}(\mathrm{ref})
\]

- reference rejected score:

\[
S_{\mathrm{ref}}^-(j)
=
\sum_{r=1}^{K_j}
\ell_{j,r}^{-}(\mathrm{ref})
\]

代码位置：
`src/g05/models/g05/g05_policy.py:757-764`。

policy margin：

\[
\Delta_\theta(j)
=
S_\theta^+(j)-S_\theta^-(j).
\]

reference margin：

\[
\Delta_{\mathrm{ref}}(j)
=
S_{\mathrm{ref}}^+(j)-S_{\mathrm{ref}}^-(j).
\]

DPO reward margin：

\[
m_j
=
\Delta_\theta(j)-\Delta_{\mathrm{ref}}(j)
=
\left(S_\theta^+(j)-S_\theta^-(j)\right)
-
\left(S_{\mathrm{ref}}^+(j)-S_{\mathrm{ref}}^-(j)\right).
\]

这表示：新 policy 相比 reference 是否更偏向 chosen。

当前代码的 DPO loss 是：

\[
L_{\mathrm{DPO}}
=
\frac{1}{B}
\sum_{j=1}^{B}
-\log\sigma(\beta m_j),
\]

其中：

- \(B\)：batch 中 pair/window 的数量。
- \(\sigma(u)=\frac{1}{1+\exp(-u)}\)。
- \(\beta=0.1\) 默认，脚本传入：
  `scripts/experiments/so101_ar_rl/train_ar_dpo.sh:94`。

代码位置：
`src/g05/models/g05/g05_policy.py:765-766`。

最终总 loss 加了 chosen CE：

\[
L
=
L_{\mathrm{DPO}}
\lambda L_{\mathrm{CE}}^+,
\qquad
\lambda=0.2.
\]

代码位置：
`src/g05/models/g05/g05_policy.py:767-768`，
脚本参数：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:95`。

\(L_{\mathrm{CE}}^+\) 是 chosen side action tokens 的平均 NLL：

\[
L_{\mathrm{CE}}^+
=
\frac{1}{\sum_{j,r}|\mathcal A_{j,r}^+|}
\sum_{j,r}
\sum_{k\in\mathcal A_{j,r}^+}
-\log \pi_\theta(z^+_{j,r,k}\mid x^+_{j,r},z^+_{j,r,<k}).
\]

它的作用是：DPO 只要求“chosen 相对 rejected 更高”，理论上可能同时降低两边概率；
加一个 chosen CE 可以防止成功轨迹本身的 action likelihood 被压低。

日志指标：

- `rl/dpo_loss`
- `rl/chosen_ce_loss`
- `rl/total_loss`
- `rl/reward_margin`
- `rl/reward_accuracy`
- `rl/policy_chosen_logp`
- `rl/policy_rejected_logp`
- `rl/ref_chosen_logp`
- `rl/ref_rejected_logp`

代码位置：`src/g05/models/g05/g05_policy.py:789-800`。

其中：

\[
\mathrm{reward\_accuracy}
=
\frac{1}{B}
\sum_{j=1}^{B}
\mathbf 1[m_j>0].
\]

代码位置：`src/g05/models/g05/g05_policy.py:769`。

## 12. DPO 梯度和 memory-light surrogate

普通 DPO 直接反传会同时保留 chosen 和 rejected 整个窗口的计算图。`WINDOW_SIZE=8`
时，一个样本包含 chosen 8 帧和 rejected 8 帧，很容易 OOM。当前训练循环用了
memory-light backward。

训练循环位置：`scripts/finetune.py:1313-1393`。

它做两步：

1. `torch.no_grad()` 下算完整窗口的 DPO margin 和一阶系数：
   `scripts/finetune.py:1319-1322`。
2. 分别对 chosen side 和 rejected side 构造 surrogate loss 反传：
   `scripts/finetune.py:1333-1392`。

现在推导为什么这样做梯度等价。

对单个样本 \(j\)：

\[
L_j
=
-\log\sigma(\beta m_j).
\]

设：

\[
u_j=\beta m_j.
\]

有：

\[
L_j=-\log\sigma(u_j).
\]

导数：

\[
\frac{\partial L_j}{\partial u_j}
=
\sigma(u_j)-1.
\]

因为：

\[
\frac{d}{du}\log\sigma(u)
=
1-\sigma(u),
\]

所以负号后：

\[
\frac{d}{du}\left[-\log\sigma(u)\right]
=
\sigma(u)-1.
\]

又因为 \(u_j=\beta m_j\)，所以：

\[
\frac{\partial L_j}{\partial m_j}
=
\beta(\sigma(\beta m_j)-1).
\]

batch mean 后：

\[
c_j
=
\frac{\partial L_{\mathrm{DPO}}}{\partial m_j}
=
\frac{\beta(\sigma(\beta m_j)-1)}{B}.
\]

代码里的：

```python
chosen_coeff = beta * (sigmoid(dpo_logits.detach()) - 1.0) / denom
rejected_coeff = -chosen_coeff
```

正是这个：
`src/g05/models/g05/g05_policy.py:770-772`。

因为：

\[
m_j
=
S_\theta^+(j)-S_\theta^-(j)
-
\Delta_{\mathrm{ref}}(j),
\]

且 reference 部分对 \(\theta\) 是常量，所以：

\[
\frac{\partial m_j}{\partial S_\theta^+(j)}=1,
\qquad
\frac{\partial m_j}{\partial S_\theta^-(j)}=-1.
\]

因此：

\[
\frac{\partial L_{\mathrm{DPO}}}{\partial S_\theta^+(j)}
=
c_j,
\]

\[
\frac{\partial L_{\mathrm{DPO}}}{\partial S_\theta^-(j)}
=
-c_j.
\]

把 \(c_j\) detach 后，构造 surrogate：

\[
\tilde L_j
=
c_j S_\theta^+(j) - c_j S_\theta^-(j).
\]

那么：

\[
\nabla_\theta \tilde L_j
=
c_j\nabla_\theta S_\theta^+(j)
-c_j\nabla_\theta S_\theta^-(j),
\]

这和原始 DPO loss 在当前参数点的一阶梯度相同。

实现位置：

- 计算系数：`src/g05/models/g05/g05_policy.py:748-787`。
- surrogate forward：`src/g05/models/g05/g05_policy.py:825-861`。
- 训练循环分侧 backward：
  `scripts/finetune.py:1333-1392`。

如果窗口太大，还会把 side 内部再按 sample 轴切成 micro-batch：
`scripts/finetune.py:1358-1389`。

chosen CE 的缩放也处理了：当 chosen side 被切块时，每块 CE 乘上
`chunk_token_sum / full_token_sum`，保证所有 chunk 加起来仍是完整 chosen CE。
代码位置：`src/g05/models/g05/g05_policy.py:845-856`。

这个机制改变的是显存调度，不改变当前参数点的一阶梯度。

## 13. DPO 脚本具体参数

`train_ar_dpo.sh` 默认：

```bash
BASE_CKPT=scripts/so101_square_finetune/runs/so100/so101_square_4ep_20260623_205729/checkpoints/step_9740.pt
DPO_SAMPLE_MODE=window
WINDOW_SIZE=8
WINDOW_STRIDE=4
WINDOW_ALIGN=relative
MAX_WINDOWS_PER_PAIR=0
LOGP_MICRO_BATCH_SIZE=1
MAX_STEPS=600
CHECKPOINTING_STEPS=200
EVAL_STEPS=100000
learning_rate=5e-6
beta=0.1
chosen_ce_weight=0.2
model.batch_size=1
model.use_pretrained_norm_stats=true
continuous_action=false
return_continuous_action=false
```

代码位置：
`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:7-23`,
`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:41-44`,
`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:53-96`。

几个关键点：

- `OVERRIDE_DATASET` 指向 `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml`：
  `scripts/experiments/so101_ar_rl/train_ar_dpo.sh:31`。
- 这个 data yaml 指定 5 个 prepared dataset dirs：
  `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml:53-60`。
- 它使用 SO100 canonical dataset adapter：
  `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml:10-13`。
- 形状是 6 维 action/state，三路相机：
  `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml:16-52`。
- processor 里用了 `RelativeJointTransform`：
  `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml:100-103`。
- `drop_high_level_prob=1.0`，也就是不保留 high-level 文本分支：
  `scripts/experiments/so101_ar_rl/configs/so101_ar_rl_data.yaml:120-123`。

如果想最接近 GRAPE 的 all sliding windows，可设置：

```bash
WINDOW_STRIDE=1 bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

如果还想最接近 GRAPE 的同 index 对齐，可再设置：

```bash
WINDOW_ALIGN=index WINDOW_STRIDE=1 bash scripts/experiments/so101_ar_rl/train_ar_dpo.sh
```

但这会显著增加 `train_samples` 和 reference cache 时间。

## 14. SFT 对照脚本

`train_ar_sft_success.sh` 和 DPO 使用同一个 base checkpoint、同一个 prepared dataset、
同一个坏 episode 过滤：

- base checkpoint：`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:9`。
- data yaml：`scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:10`。
- label root 和 exclude：
  `scripts/experiments/so101_ar_rl/train_ar_sft_success.sh:72-76`。

它只套 `SuccessFrameDataset`，不构造 pairs、不用 reference logp。训练目标取决于
`SFT_ACTION_ONLY`：

- `false`：走普通 `model(batch)`，代码位置 `scripts/finetune.py:1409-1410`。
- `true`：走 action-only SFT，代码位置 `scripts/finetune.py:1400-1408`。

action-only SFT 的 loss 是：

\[
L_{\mathrm{SFT-action}}
=
\frac{1}{N}
\sum_{i}
\sum_{k\in\mathcal A_i}
-\log \pi_\theta(z_{i,k}\mid x_i,z_{i,<k}),
\]

其中 \(N=\sum_i|\mathcal A_i|\)。实现：
`src/g05/models/g05/g05_policy.py:807-823`。

如果你要做最公平的 DPO vs SFT 对照，建议显式跑：

```bash
SFT_ACTION_ONLY=true bash scripts/experiments/so101_ar_rl/train_ar_sft_success.sh
```

这样 SFT 和 DPO 都只在 action-token 子集上优化。

## 15. GRAPE/TPO 做了什么

GRAPE README 说它包含：

- Customized Cost Generation。
- Iterative Trajectory-wise Preference Optimization。

代码位置：`D:\workspace\GRAPE\README.md:31-36`。

训练入口是 OpenVLA LoRA TPO：
`D:\workspace\GRAPE\README.md:67-93`。

### 15.1 GRAPE 的 preference 轨迹怎么来

GRAPE 在 rollout 中为轨迹打分，然后用分数排序来构造 preference。README 说 final
GCPG reward 会出现在文件名中，可以按这个 reward 排序：
`D:\workspace\GRAPE\README.md:118-120`。

在 `Data Collection/maniskill2_evaluator.py` 里，外部 cost score 的一种形式是：

\[
R_{\mathrm{ext}}
=
\prod_{g=1}^{G}
\exp\left(
-
\sum_{c=1}^{C}
\beta_c
\max(0, q_{g,c}-\tau_{g,c})
\right).
\]

变量：

- \(G\)：stage 数，代码里常用 3。
- \(C\)：constraint/cost 项数量，代码里是 `col_cost`, `grasp_cost`, `path_cost` 三项。
- \(q_{g,c}\)：第 \(g\) 个 stage 的第 \(c\) 个 cost。
- \(\tau_{g,c}\)：该 cost 的 threshold。
- \(\beta_c\)：该 cost 的权重。

代码位置：
`D:\workspace\GRAPE\Data Collection\maniskill2_evaluator.py:119-143`。

自信度 score 来自模型生成动作 token 的概率。修改版 `modeling_prismatic.py` 会把生成
动作 token 的 logprob 加起来，再取 exp：

\[
p_t
=
\exp\left(
\sum_{k=1}^{D}
\log p_\theta(a_{t,k}\mid o_t,\text{instruction},a_{t,<k})
\right).
\]

代码位置：
`D:\workspace\GRAPE\Data Collection\modeling_prismatic.py:516-541`。

rollout 时累乘：

\[
R_{\mathrm{self}}
=
\prod_t \max(p_t,0.05).
\]

代码位置：
`D:\workspace\GRAPE\Data Collection\maniskill2_evaluator.py:279-283`。

最终示例分数：

\[
R_{\mathrm{traj}}
=
0.01\log R_{\mathrm{self}}
+
R_{\mathrm{ext}}
+
5I_{\mathrm{success}}.
\]

如果 `inner_score==0`，代码走一个特殊分支：

\[
R_{\mathrm{traj}}
=
R_{\mathrm{ext}}
+
5I_{\mathrm{success}}
-1.
\]

代码位置：
`D:\workspace\GRAPE\Data Collection\maniskill2_evaluator.py:369-379`。

其中：

- \(I_{\mathrm{success}}=1\) 表示成功，否则为 0。
- \(R_{\mathrm{ext}}\) 是外部约束/cost score。
- \(R_{\mathrm{self}}\) 是模型自身动作概率累积 score。

当前 G0.5 repo 没有复刻这套 GCPG 打分。它直接使用 label 中的 `success`，并按
`instruction + init_config_id` bucket 构造 chosen/rejected。

### 15.2 GRAPE 的 window

GRAPE `window_batch` 硬编码：

```python
window_size = 8
num_windows = len(images) - window_size + 1
for i in range(num_windows):
    window = steps[i:i+8]
```

代码位置：`D:\workspace\GRAPE\TPO-Train\finetune.py:221-262`。

数学上，对一条轨迹 \(e=(s_1,\ldots,s_T)\)，它生成：

\[
W_i(e)
=
(s_i,s_{i+1},\ldots,s_{i+7}),
\qquad
i=1,\ldots,T-8+1.
\]

这是 stride 1 的 all sliding windows。

随后 `flatshape` 把 chosen/rejected window pair 保持同步打乱：
`D:\workspace\GRAPE\TPO-Train\finetune.py:511-525`。

### 15.3 GRAPE 的 DPO loss

GRAPE 的 `get_batch_logps`：

- shift labels/logits。
- 忽略 `-100`。
- 对非 mask token 的 logprob 求和。

代码位置：`D:\workspace\GRAPE\TPO-Train\finetune.py:386-413`。

`dpo_loss` 形式是：

\[
L
=
-\log\sigma\left(
\beta
\left[
(\log\pi_\theta^+ - \log\pi_\theta^-)
-
(\log\pi_{\mathrm{ref}}^+ - \log\pi_{\mathrm{ref}}^-)
\right]
\right).
\]

代码位置：`D:\workspace\GRAPE\TPO-Train\finetune.py:415-475`。

训练循环对窗口内 8 个 step 逐个算 policy/ref logp：
`D:\workspace\GRAPE\TPO-Train\finetune.py:715-824`。

然后累加：

```python
loss_step = 0.1 * (
    policy_chosen_logps
    - policy_rejected_logps
    - ref_chosen_logps
    + ref_rejected_logps
)
loss_sum += loss_step
loss_sum = -F.logsigmoid(loss_sum)
```

代码位置：`D:\workspace\GRAPE\TPO-Train\finetune.py:825-849`。

这等价于：

\[
L_{\mathrm{GRAPE-window}}
=
-\log\sigma
\left(
0.1
\sum_{r=1}^{8}
\left[
\ell_{\theta,r}^+
-\ell_{\theta,r}^-
-\ell_{\mathrm{ref},r}^+
+\ell_{\mathrm{ref},r}^-
\right]
\right).
\]

重新整理：

\[
L_{\mathrm{GRAPE-window}}
=
-\log\sigma
\left(
0.1
\left[
\left(\sum_r \ell_{\theta,r}^+ - \sum_r \ell_{\theta,r}^-\right)
-
\left(\sum_r \ell_{\mathrm{ref},r}^+ - \sum_r \ell_{\mathrm{ref},r}^-\right)
\right]
\right).
\]

这就是窗口级 DPO。

## 16. 当前 G0.5 和 GRAPE 的核心差异

| 维度 | GRAPE | 当前 G0.5 repo |
|---|---|---|
| 模型 | OpenVLA | G0.5 Qwen3.5 VLA |
| 微调方式 | LoRA-TPO | 常规 G05Policy 训练；脚本未引入 LoRA |
| 数据格式 | RLDS chosen/rejected 两套数据 | LeRobot prepared root + label/pair JSONL |
| pair 规则 | chosen 第 n 条对应 rejected 第 n 条 | 同 `instruction + init_config_id` bucket 内 success × failure |
| preference 来源 | GCPG score 排序，含 success/cost/self score | label 中 `success`，没有 GCPG cost |
| window | 长度 8，stride 1，硬编码 | 默认长度 8，stride 4，可设 relative/index/max |
| 长度不一致处理 | 基本依赖数据一一对应 | 默认 relative progress 对齐 |
| reference | 训练时同时加载 ref model | 预先/训练前 cache ref logps |
| loss | 窗口 DPO | 窗口 DPO + `0.2 * chosen CE` |
| 反传 | 直接保留窗口图 | no-grad margin + 分侧 surrogate + micro-batch |
| token 处理 | 手写 OpenVLA prompt 和 labels | G0.5 processor/BAR tokenizer，再按 action-token id 过滤 |

最重要的思想对应关系：

\[
\text{GRAPE TPO window}
\quad\leftrightarrow\quad
\text{G0.5 PreferencePairDataset sample\_mode=window}.
\]

区别在于 GRAPE 默认是 all stride-1 windows，而 G0.5 默认是 stride-4 relative windows。

## 17. TPO 和 DPO 到底是什么关系

DPO 是一个具体 loss：

\[
-\log\sigma
\left(
\beta
\left[
(\log\pi_\theta(y^+) - \log\pi_{\mathrm{ref}}(y^+))
-
(\log\pi_\theta(y^-) - \log\pi_{\mathrm{ref}}(y^-))
\right]
\right).
\]

TPO 在 GRAPE 这里不是另一个完全不同的 optimizer，而是把 \(y^+\) 和 \(y^-\) 从
“单条 response”换成“trajectory/window”：

\[
y^+
=
(a^+_t,\ldots,a^+_{t+w-1}),
\qquad
y^-
=
(a^-_{t'},\ldots,a^-_{t'+w-1}).
\]

于是：

\[
\log\pi_\theta(y^+)
=
\sum_{r=0}^{w-1}
\log\pi_\theta(a^+_{t+r}\mid o^+_{t+r},\mathrm{instruction}),
\]

\[
\log\pi_\theta(y^-)
=
\sum_{r=0}^{w-1}
\log\pi_\theta(a^-_{t'+r}\mid o^-_{t'+r},\mathrm{instruction}).
\]

当前 G0.5 repo 正是这样实现的：

- window 内每个 frame 的 action-token logprob 由 `cal_action_logps` 得到。
- `aggregate_anchor_logps` 把 window 内 frame logprob 求和。
- `compute_ar_dpo_terms` 用求和后的 window score 进 DPO。

代码链路：

```text
PreferencePairDataset.__getitem__
  -> collate_preference_pairs
  -> G05Policy.compute_ar_action_logps
  -> ARHelper.cal_action_logps
  -> aggregate_anchor_logps
  -> G05Policy.compute_ar_dpo_terms
```

对应代码：

- `src/g05/rl/so101_preference_data.py:558-630`
- `src/g05/models/g05/g05_policy.py:655-692`
- `src/g05/models/g05/helpers/ar_helper.py:301-358`
- `src/g05/rl/so101_preference_data.py:633-635`
- `src/g05/models/g05/g05_policy.py:748-787`

## 18. 可以怎么调参

最常用的调节点都在 `train_ar_dpo.sh` 的环境变量：

- `WINDOW_SIZE`
  - 越大，越像轨迹级 preference，但显存和 cache 成本越高。
  - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:14`。

- `WINDOW_STRIDE`
  - 越小，窗口越密。
  - `1` 最接近 GRAPE 的 all sliding windows。
  - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:15`。

- `WINDOW_ALIGN`
  - `relative`：按相对进度对齐，适合 chosen/rejected 长度不同。
  - `index`：同 index 对齐，最接近 GRAPE。
  - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:16`。

- `MAX_WINDOWS_PER_PAIR`
  - `0` 表示保留 stride 规则下全部窗口。
  - 大于 0 表示每个 base pair 最多保留这么多个窗口。
  - 代码位置：`src/g05/rl/so101_preference_data.py:407-411`。

- `LOGP_MICRO_BATCH_SIZE`
  - 控制 action logp / surrogate backward 的 sample 轴切分。
  - 默认 1 是最省显存。
  - 代码位置：`scripts/experiments/so101_ar_rl/train_ar_dpo.sh:18`,
    `scripts/finetune.py:1324-1389`。

- `rl.beta`
  - DPO 温度。
  - 大会让 margin 的惩罚更陡，偏好信号更强但可能不稳。
  - 当前默认 0.1。

- `rl.chosen_ce_weight`
  - chosen CE 正则权重。
  - 当前默认 0.2。

## 19. 一段伪代码总结

```python
# 1. prepare data
for raw_run in raw_root:
    prepared_run = transform_to_g05_model_frame(raw_run)
    rewrite_labels_dataset_dir(prepared_run)

# 2. build base pairs
pairs = []
for bucket in group_by(labels, key=(instruction, init_config_id)):
    for success_episode in bucket.success:
        for failure_episode in bucket.failure:
            pairs.append((success_episode, failure_episode))

# 3. expand base pairs into window samples
items = []
for pair in pairs:
    chosen_range = episode_index[pair.chosen]
    rejected_range = episode_index[pair.rejected]
    for chosen_start, rejected_start, w in window_start_pairs(...):
        items.append({
            "chosen_indices": chosen_start : chosen_start + w,
            "rejected_indices": rejected_start : rejected_start + w,
        })

# 4. cache reference logps once
for item in items:
    ref_chosen = sum(logp_ref(action_tokens(frame)) for frame in item.chosen)
    ref_rejected = sum(logp_ref(action_tokens(frame)) for frame in item.rejected)
    write_cache(item.pair_id, ref_chosen, ref_rejected)

# 5. train
for batch in dataloader(items):
    policy_chosen = sum(logp_theta(action_tokens(frame)) for frame in batch.chosen)
    policy_rejected = sum(logp_theta(action_tokens(frame)) for frame in batch.rejected)
    ref_chosen, ref_rejected = read_cache(batch.pair_id)

    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    dpo_loss = -log_sigmoid(beta * margin).mean()
    chosen_ce = mean_nll(chosen_action_tokens)
    loss = dpo_loss + chosen_ce_weight * chosen_ce
    backward(loss)  # implemented by equivalent memory-light surrogate
```

## 20. 最后要记住的三个点

第一，当前实验的核心不是“让 reward 直接回归到动作”，而是让模型在相同任务/相似初始
状态下，相对 reference 更偏向成功轨迹的 action-token 序列。

第二，window DPO 的关键是把多个 frame 的 logprob 求和后再做 DPO。它不是对窗口内
每一帧独立做 8 个 DPO loss；GRAPE 和当前 G0.5 代码都是先 sum，再
`-log sigmoid`。

第三，G0.5 的实现比 GRAPE 工程上更保守：reference logp 预缓存、relative window
对齐、chosen CE 正则、surrogate backward 都是为了适配 G0.5 模型和这批 SO101 数据。
