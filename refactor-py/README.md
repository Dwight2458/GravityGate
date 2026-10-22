# GravityGate Python

独立的 Python 实现，运行时不依赖 Rust。功能覆盖现有项目的 CLI、OAuth、账号池、
模型解析、OpenAI Chat Completions / Responses、SSE、签名缓存、验证恢复、
审计、Prometheus 和原有仪表盘。

采用 Python 3.12+、FastAPI、httpx、Uvicorn，使用 uv 管理依赖。
账号和运行状态重新设计为 SQLite；原 Rust 项目及其数据不受影响。

## 启动

在项目根目录执行：

```powershell
cd refactor-py
uv sync --locked
uv run gravitygate-py config init
uv run gravitygate-py account login
uv run gravitygate-py probe --model gemini-3.8-flash-high
uv run gravitygate-py serve
```

默认地址为 `http://127.0.0.1:18081`，OpenAI 客户端 Base URL 为
`http://127.0.0.1:18081/v1`。访问根路径打开仪表盘。这个端口与原服务的
`18080` 分开，可用于并行对比。更换端口：

```powershell
uv run gravitygate-py serve --port 18082
```

也可以留在项目根目录，通过 `uv run --directory refactor-py gravitygate-py ...`
执行命令。无需替换 PATH 中的 Rust `gravitygate`。

## 迁移账号

默认数据目录：Windows 为 `%APPDATA%\gravitygate-py`；Linux/macOS 为
`${XDG_CONFIG_HOME:-~/.config}/gravitygate-py`。其中 `config.toml` 保存配置，
`gravitygate.db` 保存账号、限流、验证租约和审计记录。数据库包含刷新令牌，
应按凭据文件管理；访问令牌只缓存在内存中。

Python 版不会自动读取 Rust 的账号文件。需要沿用账号时，显式导入：

```powershell
uv run gravitygate-py account import-rust "$env:APPDATA\gravitygate\accounts.json"
uv run gravitygate-py account list
```

导入保留已有验证、配额和禁用状态，不继承 Rust 进程正在持有的检查租约。
相同刷新令牌对应的已有 Python 账号不会被覆盖；原 JSON 文件保持原样。
配置不自动导入，可参考 [examples/config.toml](examples/config.toml) 手动设置。

使用独立测试目录时，将全局参数放在命令前：

```powershell
uv run gravitygate-py --data-dir .\local-data config init
uv run gravitygate-py --data-dir .\local-data account list
```

环境变量 `GRAVITYGATE_PY_DATA_DIR`、`GRAVITYGATE_PY_CONFIG_FILE` 分别指定目录和
配置文件。`GG_HOST`、`GG_PORT`、`GG_API_KEYS`、`GG_LOG_LEVEL`、
`GG_ACCOUNT_STRATEGY` 覆盖 TOML；修改配置后需重启服务。

## 常用命令

```powershell
uv run gravitygate-py account login --no-open
uv run gravitygate-py account login --no-browser
uv run gravitygate-py account add-token --stdin --email you@example.com
uv run gravitygate-py account list --json
uv run gravitygate-py account verify 0
uv run gravitygate-py account recheck 0
uv run gravitygate-py account disable 0
uv run gravitygate-py account enable 0
uv run gravitygate-py account clear-holds 0
uv run gravitygate-py account remove 0
uv run gravitygate-py models
uv run gravitygate-py models --live
uv run gravitygate-py probe --model gemini-3.8-flash-high --repeat 2 --raw
uv run gravitygate-py config path
uv run gravitygate-py config show
```

`models` 离线列出静态目录，`models --live` 合并 Google 的实时列表。
模型名后缀、显式 thinking budget、reasoning effort、默认档位按此顺序解析。
未知模型名直接透传，例如不会把 `gemini-3.8-flash-tiered-high` 猜测成另一个模型。

`account verify` 仅验证 OAuth 与项目发现；`account recheck` 调用运行中的
Python 服务，并通过真实生成请求确认恢复。`clear-holds` 是本地人工重置，
不构成上游已恢复的证据。复检命令、服务必须使用相同配置和数据目录。

## HTTP 接口

| 接口 | 功能 |
| --- | --- |
| `POST /v1/chat/completions` | Chat，流式或普通 JSON |
| `POST /v1/responses` | Responses，流式或普通 JSON |
| `GET /v1/models` | 静态目录与缓存的实时目录 |
| `GET /health` | 账号状态及验证时间；全部不可用时返回 503 |
| `GET /account-limits` | 本地限流组及实时模型配额 |
| `POST /api/accounts/{selector}/recheck` | 固定账号立即复检 |
| `POST /refresh-token` | 清除访问令牌、项目、模型及签名等内存缓存 |
| `GET /api/stats`、`/api/stats/accounts` | 最近一小时统计 |
| `GET /api/requests?limit=40` | 最近请求记录 |
| `GET /metrics` | Prometheus 文本 |
| `GET /` | 仪表盘 |

Chat / Responses 默认接受最多 16 MiB 的 JSON，包括图片 data URI。
超限在选择账号前返回 413，错误码 `request_body_too_large`；该字节限制与模型的
token 上下文窗口独立。远程图片 URL 与上游不支持的工具类型沿用现有转换规则。

配置 `server.api_keys` 后所有接口均要求 Bearer 或 `x-api-key`。浏览器访问
仪表盘时支持 HTTP Basic 提示，用户名任意，密码填写网关 API key。
`config show`、账号列表、审计记录均不输出刷新令牌/API key。

Responses 无服务端对话存储：后续轮次应在 `input` 中带回历史，
`previous_response_id` 返回明确错误。工具调用 ID 和思考签名在进程内缓存，
重启后 Gemini 使用现有哨兵回退规则，Claude 不重放跨模型族签名。

## 限流与验证恢复

- 配额按模型组保存，Flash 的档位共享限流，不阻塞 Pro/Claude；模型容量不足
  不会写成账号配额耗尽，也不降低账号健康分。
- 账号选择与令牌扣减在一个 SQLite 事务内完成。提供 hybrid、sticky、
  round-robin、least-recently-used 策略；hybrid 保留加权评分和切换阈值。
  低健康分通过评分和有限冷却处理，不会成为永久不可用的独立开关。
- 自动检查只针对待验证账号，默认 15 秒扫描。失败按 1/2/5 分钟退避，真实上游
  配额等待可更长；正常账号不会被持续探测。
- 恢复要求非空白可见文本、`STOP` 和完整流结束；200、纯思考、MAX_TOKENS、
  断流及错误事件均不算恢复。
- 检查有持久化租约，重复启动 15 秒内返回 409。超时/取消会释放租约，进程崩溃
  后租约在超时加 30 秒到期；人工重置、新挑战或账号删除会使旧结果失效。
- 网页验证必须使用需要验证的 Google 账号。完成后可点仪表盘 **Recheck now**，
  或等待自动检查；成功会清除验证链接并恢复健康分，无需重启。

## 开发与验证

```powershell
uv sync --locked
uv run pytest -q
uv run ruff check src tests tools/generate_contracts.py
uv run ruff format --check src tests tools/generate_contracts.py
uv build
```

测试使用合成账号、本地假上游和回环 TCP。覆盖两个协议的流式/普通响应、
OAuth PKCE 回调、工具签名重放、账号切换、并发写入、配额分类、413、验证竞态、
审计和真实 HTTP 增量传输，不读取真实账号、不消耗 Google 配额。

`tests/fixtures/rust-contracts.json` 保存 100 组从原 Rust 代码离线导出的模型、
Schema 和协议转换预期。正常 Python 测试不需要 Rust。维护对照样本时才运行：

```powershell
uv run python tools/generate_contracts.py
```

`tools/oracle` 仅为开发期对照工具，不属于 Python 运行依赖。源码、wheel 中均
包含独立仪表盘；实现边界见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

当前验证针对本地行为；未使用真实 Google 账号验证 Python 传输。实际 Google
接入需由操作者显式导入/登录后运行 `probe`，本地 HTTP 一致性不等于上游已接受。
