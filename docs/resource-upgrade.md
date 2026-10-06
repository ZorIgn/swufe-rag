# 正文、工具资源与并发会话

HTTP 在读取正文前获取请求并发额度，逐块统计正文大小，越界立即返回 413；读取超时返回 408 并归还额度。`SWUFE_BODY_TIMEOUT_SECONDS` 默认 10 秒。既有 `SWUFE_REQUEST_MAX_BYTES` 继续控制大小。

工具改用进程共享的有界执行器，`SWUFE_TOOL_WORKERS` 默认 8。没有无界任务队列；运行中任务即使调用方已超时，也保留额度，直到 future 实际结束。容量耗尽返回 typed `tool_capacity_exceeded`，不会创建更多线程。线程无法强制终止，工具自身的网络/数据库超时仍有必要。

会话开始时登记唯一请求令牌，结束时按令牌条件更新；较早请求完成不能覆盖较晚请求的上下文。内存存储在锁内校验，Redis 通过 Lua 原子校验和写入，覆盖多 worker。

回归：`python -m pytest -q tests/canonical`。安装项目 `redis` extra 并设置 `SWUFE_TEST_REDIS_URL`，可运行真实 Redis 的双实例条件更新测试；默认无此环境变量时仅跳过该集成检查。测试创建独立临时命名空间，并清理自身键。
