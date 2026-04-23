#!/bin/bash
# =============================================================
# init_repo.sh — 项目 Git 初始化 + GitHub 上传引导
# 在项目根目录运行: bash scripts/init_repo.sh
# 两个项目各运行一次，流程完全相同
# =============================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

PROJECT_NAME=$(basename "$PWD")
echo -e "${CYAN}▶ 正在初始化项目：$PROJECT_NAME${NC}"
echo ""

# ── Step 1: git 初始化 ────────────────────────────────────────
git init
git branch -M main
echo -e "${GREEN}✅ git init 完成${NC}"

# ── Step 2: 生成 .gitignore ───────────────────────────────────
cat > .gitignore << 'EOF'
# 模型权重（体积太大，不上传）
*.bin
*.pt
*.pth
*.ckpt
*.safetensors
*.h5

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# 环境变量
.env
.env.*

# 实验追踪工具缓存
wandb/
runs/
lightning_logs/
.neptune/

# 视频（体积大）
*.mp4
*.avi
*.mov

# Jupyter
.ipynb_checkpoints/

# 系统文件
.DS_Store
Thumbs.db

# 编辑器
.vscode/
.idea/

# 原始数据集（不上传）
data/raw/
EOF
echo -e "${GREEN}✅ .gitignore 已生成${NC}"

# ── Step 3: 创建必要目录 ──────────────────────────────────────
mkdir -p logs scripts reports dataset
touch logs/.gitkeep reports/.gitkeep

# ── Step 4: 生成 README.md ────────────────────────────────────
if [ ! -f README.md ]; then
cat > README.md << EOF
# $PROJECT_NAME

> 本项目为机器人学习相关研究记录，包含实验代码、配置文件与实验日志。

## 环境

\`\`\`bash
conda activate vla_env  # Python 3.10, torch 2.7.1+cu118
\`\`\`

## 目录结构

\`\`\`
$(basename $PWD)/
├── CLAUDE.md      # Claude Code 上下文文件
├── README.md      # 本文件
├── logs/          # 实验日志（手动 Markdown 记录）
├── reports/       # 对比报告与总结
└── scripts/       # 工具脚本
\`\`\`

## 实验记录

详见 \`logs/\` 目录，每次实验对应一个 \`.md\` 文件。

## 主要结论

（随实验进展持续更新）
EOF
echo -e "${GREEN}✅ README.md 已生成${NC}"
fi

# ── Step 5: 首次提交 ──────────────────────────────────────────
git add -A
git commit -m "init: project scaffold with CLAUDE.md, README and gitignore"
echo -e "${GREEN}✅ 首次 commit 完成${NC}"

# ── Step 6: GitHub 上传引导 ───────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  接下来：把项目上传到 GitHub（按步骤操作）         ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}【第一步】在 GitHub 网站上新建仓库${NC}"
echo "  1. 打开 https://github.com/new"
echo "  2. Repository name 填写：$PROJECT_NAME"
echo "  3. 选 Private（私有）或 Public（公开）"
echo "  4. ⚠️  不要勾选任何初始化选项（README/gitignore 都不选）"
echo "  5. 点击「Create repository」"
echo ""
echo -e "${YELLOW}【第二步】回到服务器终端，运行以下命令${NC}"
echo "  （把 YOUR_USERNAME 替换成你的 GitHub 用户名）"
echo ""
echo -e "  ${GREEN}git remote add origin https://github.com/YOUR_USERNAME/$PROJECT_NAME.git${NC}"
echo -e "  ${GREEN}git push -u origin main${NC}"
echo ""
echo -e "${YELLOW}【第三步】首次 push 会要求登录${NC}"
echo "  - 用户名：你的 GitHub 用户名"
echo "  - 密码：⚠️ 不是账号密码！要用「Personal Access Token」"
echo "  - 获取 Token：GitHub → Settings → Developer settings"
echo "    → Personal access tokens → Tokens (classic) → Generate new token"
echo "    → 勾选 repo 权限 → 生成后复制（只显示一次！）"
echo ""
echo -e "${YELLOW}【后续每次提交】${NC}"
echo "  git add -A"
echo "  git commit -m \"exp: 描述本次实验\""
echo "  git push origin main"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ 本地初始化全部完成，按上方步骤连接 GitHub 即可${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"