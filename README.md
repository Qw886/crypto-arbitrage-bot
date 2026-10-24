# crypto-arbitrage-bot

基于 Python 的加密货币套利研究与交易系统。当前代码包含现货/永续研究、成本核算、
行情质量检查、受控测试网执行、双腿订单状态管理、对账和异常恢复。

**尚未证明策略具有可持续的独立净收益；本仓库不承诺盈利。**
测试通过表示被覆盖的行为符合测试预期，不代表未来收益或实盘安全资格。

## 这份源码副本包含什么

- `src/crypto_arbitrage_bot/`：产品源码，包括基于 Decimal 的计算与 UTC 时间处理。
- `tests/`：离线自动化测试、公开行情样例和网络/凭据访问保护。
- `work/`：产品或测试实际依赖的少量辅助脚本，保留路径以免破坏导入。
- `docs/*.toml`：既有测试使用的配置样例；历史窗口和开关不是当前运行指令。

本副本未包含本机数据库、真实凭据、日志、备份、全部研究工具或历史验收收据，
不应据此声称可以复现原工作区的全部研究结论。Hummingbot 原型及下载副本未纳入。

## 本地查看和离线测试

项目声明支持 Python 3.12 或更新版本。本轮实际验证 Windows / CPython 3.13.15
的全新虚拟环境安装，未冒称已经实测其他系统或 Python 版本。
下面的安装会访问包仓库；测试步骤使用本地假数据和网络访问保护。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

在项目根目录执行测试。完整测试目前依赖 Windows D 盘；一组独立进程恢复测试明确要求
临时目录位于 `D:\CodexValidation` 内，因此不能随意换用其他根目录并声称已通过完整验收。
这是当前可移植性限制，不能删除路径保护来绕过。
`--basetemp` 会清理已有目标，所以只使用经过检查不存在的新目录。

```powershell
$validationRoot = [IO.Path]::GetFullPath('D:\CodexValidation')
New-Item -ItemType Directory -Path $validationRoot -Force | Out-Null
$validationTarget = Join-Path $validationRoot ('github-tests-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $validationTarget) { throw 'Test target already exists' }
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp $validationTarget
.\.venv\Scripts\python.exe -B -m mypy --no-incremental
.\.venv\Scripts\python.exe -B -m ruff check --no-cache tests
.\.venv\Scripts\python.exe -B -m ruff check --no-cache src tests
```

测试保护在收集阶段阻止已列明的 Python 网络访问和已知真实凭据路径，
并向 Python 子进程传递保护。这不是面向恶意代码的操作系统沙箱。
详细边界见 [离线测试说明](docs/offline-test-safety.md)。

## 运行边界

源码中存在测试网执行入口，不能把“有入口”理解为已获准执行。
不要向这份副本添加实际密钥、恢复耗尽的测试预算或依据历史配置发单。
`.env.example` 仅展示研究配置和假设费率，不是账户真实费用。

全范围 Ruff 检查仍有 14 项产品代码提示，会返回非零；测试代码静态检查已经通过。
详情见 [已知问题](docs/KNOWN_ISSUES.md)。没有用忽略规则隐藏这些提示。

策略独立净收益、真实费用、完整现场风险验收及其他平台可复现性仍需分别验证。
代码、自动化测试、公共市场证据、实际执行结果与策略净优势应分别报告。

## 开发说明与代码来源

项目由作者主导需求、策略方向和开发推进，使用 AI 辅助编写、检查与测试。
具体第三方代码来源尚未完成逐段溯源，当前没有据此声明所有代码均为独创。
运行依赖包含 Pydantic 和 pydantic-settings，开发依赖包含 pytest、Ruff、mypy。

本项目采用 [MIT 许可证](LICENSE)，Copyright (c) 2026 Qw886。
第三方材料仍保留各自适用的许可，来源核查范围见下文。
依赖标识和来源检查的范围见 [第三方材料说明](docs/github-third-party-notes.md)。
传递依赖见 [环境依赖记录](docs/DEPENDENCY_INVENTORY.md)，源码核查范围见
[来源说明](docs/SOURCE_PROVENANCE.md)。

## 本次发布候选验收

全新 Windows / CPython 3.13.15 环境中，完整离线测试 **4594 项通过**，mypy 与 pip check 通过。
测试代码静态检查通过；产品代码仍有已说明的 14 项提示。
详情见 [验收结果](docs/RELEASE_ACCEPTANCE.md)。此结果不证明盈利，本版已公开发布。
