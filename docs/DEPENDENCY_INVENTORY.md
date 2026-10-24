# 最终新环境依赖记录

验证平台：Windows / CPython 3.13.15。以下为本次全新环境安装的直接与传递依赖，
来源为安装分发包元数据，未将这些第三方安装包或虚拟环境加入源码包。

| 包 | 版本 | 元数据许可证标识 |
| --- | --- | --- |
| annotated-types | 0.8.0 | MIT |
| ast_serialize | 0.13.1 | MIT |
| colorama | 0.4.6 | BSD License |
| iniconfig | 2.3.1 | MIT |
| librt | 0.16.0 | MIT |
| mypy | 2.3.1 | MIT |
| mypy_extensions | 1.1.0 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pathspec | 1.1.1 | Mozilla Public License 2.0 (MPL 2.0) |
| pluggy | 1.7.0 | MIT |
| pydantic | 2.13.5 | MIT |
| pydantic-settings | 2.15.0 | MIT |
| pydantic_core | 2.46.5 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| ruff | 0.16.7 | MIT |
| setuptools | 84.0.0 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |

本记录包含开发验收需要的 setuptools==84.0.0；pip 属于环境安装工具，未列入应用依赖表。
这是成功安装环境的版本记录，不是所有系统通用的依赖锁。元数据标识不替代许可证全文或合规认证。
