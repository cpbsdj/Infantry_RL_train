# Infantry — Wheel-Leg Robot RL

基于 [mjlab](https://github.com/mujocolab/mjlab) 的轮腿机器人强化学习训练环境。

## 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器
- NVIDIA GPU
- MuJoCo（由 mjlab 自动安装）

## 快速开始

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆仓库并安装依赖

```bash
git clone <your-repo-url> Infantry_test
cd Infantry_test
uv sync
```

这会自动创建虚拟环境并安装所有依赖（包括 mjlab 和 infantry 包本身）。

### 3. 验证任务注册

```bash
uv run list-envs
```

应能看到 `Mjlab-Velocity-Flat-Infantry` 和 `Mjlab-Velocity-Rough-Infantry`。

### 4. 配置 W&B 实验追踪

训练过程会通过 [Weights & Biases](https://wandb.ai) 记录实验数据。首次使用前需要完成以下配置：

#### 4.1 注册并登录 wandb

```bash
pip install wandb
wandb login
# 输入你的 API Key（从 https://wandb.ai/authorize 获取）
```

#### 4.2 设置 Entity（Team）

实验数据需要上传到团队 `wheelleg_RL_train_group` 下。请通过以下**任一方式**设置：

**方式一：终端设置环境变量（推荐）**

每次训练前在终端中执行：

```bash
export WANDB_ENTITY="wheelleg_RL_train_group"
```

也可以写入 `~/.bashrc` 永久生效：

```bash
echo 'export WANDB_ENTITY="wheelleg_RL_train_group"' >> ~/.bashrc
source ~/.bashrc
```

**方式二：在 wandb 网页端设置默认 Entity**

访问 [wandb.ai/settings](https://wandb.ai/settings) → 找到 **"Default location to create new projects"** → 设置为 `wheelleg_RL_train_group`。

> **注意**：如果不设置 `WANDB_ENTITY`，实验数据会默认上传到个人账户下，团队成员无法查看。

### 5. 训练

```bash
# 单 GPU 训练（Flat 地形）
uv run train Mjlab-Velocity-Flat-Infantry

# 多 GPU
uv run train Mjlab-Velocity-Flat-Infantry --gpu-ids 0,1
```

训练日志和 checkpoint 保存在 `logs/rsl_rl/<experiment_name>/<timestamp>/` 下。

### 6. 使用 MuJoCo Simulator 可视化回放

```bash
# 找到最新的 checkpoint 路径，例如：
# logs/rsl_rl/infantry_velocity/<timestamp>/model_xxxx.pt

# 默认 viewer（有显示器用 native，无显示器自动切 viser）
uv run play Mjlab-Velocity-Flat-Infantry \
  --checkpoint-file logs/rsl_rl/infantry_velocity/<timestamp>/model_xxxx.pt

# 指定渲染的机器人数量
uv run play Mjlab-Velocity-Flat-Infantry \
  --checkpoint-file logs/rsl_rl/infantry_velocity/<timestamp>/model_xxxx.pt \
  --num-envs 1
```

### 7. 使用 Viser 可视化回放

Viser 是基于浏览器的交互式可视化工具，适合在无显示器的远程服务器上使用。

```bash
uv run play Mjlab-Velocity-Flat-Infantry \
  --checkpoint-file logs/rsl_rl/infantry_velocity/<timestamp>/model_xxxx.pt \
  --viewer viser \
  --num-envs 1
```

启动后终端会显示 Viser 服务地址（默认 `http://0.0.0.0:8080`）。

#### 本地机器访问

- **服务器在本地**：直接在浏览器打开 `http://localhost:8080`

- **服务器在远程**：通过 SSH 端口转发访问。在本地机器另开终端执行：

  ```bash
  ssh -L 8080:localhost:8080 <user>@<server-ip>
  ```

  然后在本地浏览器打开 `http://localhost:8080`

#### Viser 界面操作

| 功能 | 说明 |
|------|------|
| 播放/暂停 | 左侧 GUI 面板的 Play/Pause 按钮 |
| 切换 checkpoint | GUI 面板列出同目录下所有 `.pt` 文件，点击即可热切换 |
| 视角控制 | 鼠标拖拽旋转，滚轮缩放，右键平移 |
| 显示所有环境 | GUI 勾选 "Show all envs"（`--num-envs > 1` 时） |
| Debug 可视化 | GUI 勾选 "Debug vis" 显示箭头/ghost mesh 等 |

## 项目结构

```
src/infantry/
├── tasks/velocity/
│   ├── config/infantry/      # 环境配置（env_cfgs, rl_cfg, rough/flat 变体）
│   ├── mdp/                   # 自定义 MDP 组件（observations, rewards）
│   └── rl/                    # RL runner 注册
└── assets/                    # 机器人模型（MJCF/XML + 网格文件）
```