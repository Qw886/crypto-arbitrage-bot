# 已知限制与静态检查说明

## 当前使用范围

- 策略尚未证明具有可持续的独立净收益，不能作为自动投入真实资金的依据。
- 原受控测试预算已经耗尽；源码、历史配置或离线测试通过不授予新的订单预算。
- 本副本的完整测试依赖 Windows，以及 `D:\CodexValidation` 下经过确认不存在的新临时目录。
  其中十个独立进程恢复用例的辅助程序明确检查该根目录。
- 安装/测试仅实测本轮 Windows / CPython 3.13.15 环境；其他平台及 Python 版本未完成验收。
- 研究/验收配置和部分辅助程序使用相对路径，按 README 从项目根目录运行。
  可编辑安装的成功不代表任意工作目录或单独 wheel 已通过全部功能验收。
- 本副本只含产品与正式测试的必要文件，不含全部历史研究工具、运行数据与审计证据。
- 网络保护针对已列明的 Python 审计事件及已知凭据路径，不是恶意代码安全沙箱。
- 作者已选择 MIT 许可证，见仓库根目录 LICENSE。

## 全范围 Ruff 的 14 项既有提示

测试代码的静态检查已通过。产品代码保留以下提示，全范围检查仍返回非零：

| 规则 | 数量 | 位置/原因 |
| --- | --- | --- |
| B905 | 2 | arbitrage/execution.py：盘口相邻档位比较使用 `zip(levels, levels[1:])`，未显式声明 strict |
| UP047 | 2 | arbitrage/sizing.py 与 utils/decimal_math.py：使用 ParamSpec/TypeVar 的既有泛型写法 |
| UP035 | 1 | exchanges/endpoints.py：Mapping 从 typing 导入 |
| UP042 | 1 | exchanges/endpoints.py：既有 `str, Enum` 类写法 |
| UP040 | 8 | models/market.py 与 exchanges/http_client.py：既有 TypeAlias 写法 |

B905 的两处迭代器有意相差一个元素，用于比较相邻价格。不能直接改成 `strict=True`，
否则正常盘口也会因长度不一致被拒绝；后续可单独审核显式 `strict=False` 的整理。

其余提示建议较新语法，不是本轮发现的计算错误。枚举字符串行为与运行时类型别名
可能影响调用方，不能将批量 unsafe-fixes 当作无行为变化整理。
本轮保留产品实现，不改金融公式、策略、风控或其冻结证据，也不关闭上述检查规则。
