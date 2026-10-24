# 离线测试凭据与网络边界

更新：2026-10-14。本阶段修复全量审计F-02及其测试设施，不修改产品金融、交易、恢复或准入源码，不授予任何新的真实执行权限。

## 现在的行为

默认在项目目录执行`.venv/Scripts/python.exe -m pytest -q`，`tests/conftest.py`会在收集阶段安装测试专用保护。原CLI检查用例显式使用临时不存在的凭据路径，避免在替换假客户端之前加载实际凭据。

- CPython的连接、DNS解析、UDP发送等已列明审计事件抛出固定PermissionError；测试中的socket模拟不会取消该审计检查。
- 已知实际项目`.env`及data/testnet目录中的DPAPI文件禁止打开；Windows大小写别名也被检查。测试以临时假文件验证，实际凭据不参与回归。
- 测试Python启动时按名称删除账户密钥、执行模式和订单开启变量，不读或打印值。更改限于本次测试进程及其子进程，不修改系统永久环境变量。
- 子进程即使显式传入环境或替换PYTHONPATH，也会在原Popen构造边界补入测试启动目录；保持原命令、工作目录、创建标志和Popen类型。孙进程同样接续。
- Python启动参数-S/-I/-E及相应组合会跳过或忽略保护，因此在真正启动前拒绝。合法的-s/-W/-X及脚本自己的-S参数保留。
- Windows命令行使用CommandLineToArgvW解析。最初使用shlex导致合法嵌套引号被拒绝，已由真实独立进程回归复现并修正；失败日志保留。

测试设施放在`tests/offline_runtime/sitecustomize.py`，产品启动入口不导入它。不要把此目录加入真实机器人运行的长期环境变量。

## 已验证范围与限制

20项专项验证默认父进程、启动保护、显式环境、孙进程、账户环境清除、Windows.env大小写别名、Python参数及实际命令行；网络反例调用公开sys.audit事件，没有尝试连接交易所或读取实际密钥。

相关368项通过（33.92秒），包含原凭据、CLI、时钟启动器、研究控制、测试网控制、原生身份、进程探测及真实终态重启。完整默认设施测试4468通过（449.08秒、0失败/警告），没有临时审查插件。mypy156源码/辅助通过，4修改或新增测试文件Ruff通过；全量仍14产品/31测试历史提示。结果与证据登记于PROJECT_STATUS顶部和work/offline-test-safety-20261014/acceptance.json。

初版相关366通过/1失败属于被后续修正替代的版本，原因是新增设施的Windows引号解析。另加的简短嵌套引号用例在旧解析器上已通过，不能单独当作红灯证据；真正红灯来自原独立控制进程回归。最终368及4468对应原生解析修正后的版本，早期日志全部保留。

这是可信测试代码的隔离措施。Python官方明确说明Python层audit hook不是恶意代码安全沙箱；原生扩展、非Python程序、未列明访问方式和启动早期第三方代码不能据此获得全面安全保证。涉及不可信代码仍需OS级隔离。依据[Python 3.13审计接口](https://docs.python.org/3.13/library/sys.html#sys.addaudithook)。

Windows引号处理和内存释放依据[Microsoft CommandLineToArgvW](https://learn.microsoft.com/en-us/windows/win32/api/shellapi/nf-shellapi-commandlinetoargvw)。本轮实际验证为本机CPython3.13.15/Windows，不冒称跨全部Python/操作系统验收。

## 容量与运行方法

审查发现C盘可用空间不足，完整验证使用D盘独立新建临时目录。D盘与C盘同一物理盘，此安排解决测试临时文件容量，不构成外部备份。

pytest的--basetemp会清理已有目标，因此只使用经过确认不存在的新目录。可按以下方式生成目标：

```powershell
$testValidationRoot = [IO.Path]::GetFullPath('D:\CodexValidation\crypto-arbitrage-bot')
New-Item -ItemType Directory -Path $testValidationRoot -Force | Out-Null
$testValidationTarget = [IO.Path]::GetFullPath((Join-Path $testValidationRoot ([guid]::NewGuid().ToString('N'))))
if (-not $testValidationTarget.StartsWith($testValidationRoot+'\', [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $testValidationTarget)) {
    throw 'Unsafe or existing test target'
}
.venv/Scripts/python.exe -m pytest -q --basetemp $testValidationTarget
```

本阶段没有删除旧测试目录、历史证据、备份或实际数据库。需要释放空间时先明确识别数据归属；不能清理原研究失败记录以制造完整覆盖。

## 交付边界

修改原tests/conftest.py和一个CLI测试，新增测试启动辅助与专项。交易订单回执语义、控制器接管/状态发布等审计缺口仍待后续处理；本阶段测试设施通过不证明这些问题已修复。

原实际20/20账本、公共研究固定期限及所有失败/NULL保持；没有实际私有请求、订单、账户核查或策略/工程资格发放。测试源代码修改不会替代完整现场风险证明和独立净优势。
