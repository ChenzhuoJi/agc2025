# agc2025

> 本项目基于 Python，采用 [uv](https://github.com/astral-sh/uv) 进行依赖管理，支持本地 `.venv` 虚拟环境。请根据以下指引配置和运行本项目。

---

## 环境准备

1. **检查 Python 版本**本项目建议使用与 `.python-version` 文件指定一致的 Python 版本（例如 Python 3.11）。检查你的 Python 版本：

   ```bash
   python --version
   ```
2. **创建虚拟环境**推荐使用 `.venv` 作为虚拟环境目录。

   使用 venv（Python 3.3+）:

   ```bash
   python -m venv .venv
   ```

   或使用 virtualenv（提前 `pip install virtualenv`）:

   ```bash
   virtualenv .venv
   ```

   激活虚拟环境：

   - **Windows**:
     ```bash
     .\.venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
3. **安装项目依赖**本项目推荐使用 uv 管理依赖。安装方法如下（任选其一）：

   - 推荐方式（根据 `pyproject.toml` 和 `uv.lock` 安装锁定的依赖）:

     ```bash
     uv sync
     ```
   - 或仅安装 `pyproject.toml` 里的依赖：

     ```bash
     uv pip install -e .
     ```
   - 如未安装 uv，可通过 pip 安装 uv:

     ```bash
     pip install uv
     ```
4. **验证依赖安装**

   ```bash
   pip list
   ```

   应显示项目所需依赖。

---

## 快速上手与运行

1. **运行实验脚本或主程序**

   ```bash
   python experiment_script.py 数据集名 预测方法
   ```

   - 例如：
     ```bash
     python experiment_script.py cora lambda
     ```
   - 其中 `cora` 是数据集名称（如 `cora`、`citeseer` 等），`lambda` 是预测方法（如 `lambda`、`communitude` 等）。
2. **可选参数和扩展配置**

   你可以通过命令行传递自定义参数。例如：

   ```bash
   python experiment_script.py cora lambda --order 6 --decay 0.8 --interWeight 4 --pairwiseWeight 3
   ```

   参数作用请参考源代码/脚本说明。

---

## 实验结果与日志

- 实验运行结束后，日志通常保存在 `results/logs/` 下，文件名包含数据集与时间戳。例如：
  ```
  results/logs/cora_20231104_1430.json
  ```
- 通过此日志文件可以分析实验结果。

---

## 项目地址与维护者

- 作者主页: [ChenzhuoJi](https://github.com/ChenzhuoJi)
- 项目地址: [https://github.com/ChenzhuoJi/agc2025](https://github.com/ChenzhuoJi/agc2025)
