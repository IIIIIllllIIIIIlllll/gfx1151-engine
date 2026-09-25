# 并发请求（PARALLEL）

面向轻负载的并发：几条短任务（知识问答、小改写）同时进来时不必排队等前一条生成完。
它不是批量 decode：GPU 同一时刻只算一条序列，各序列按轮转交替执行，所以总吞吐基本
不变（单条 decode 约 31 tok/s @7K），收益是后来的请求不用等先来的请求全部结束。

## 配置

`service.conf`：

| 项 | 默认 | 说明 |
|---|---|---|
| `PARALLEL` | 4 | 同时运行的请求数，1–8。1 = 旧的单路行为 |
| `KV_PAGED` | 1 | 必须为 1（`PARALLEL>1` 时启动器会拒绝 0） |
| `KV_POOL_TOKENS` | 0 | 共享 KV 页池大小，取 `max(KV_POOL_TOKENS, MAX_CONTEXT)`；默认即一条 256K |

引擎读 `GDEC_PARALLEL`，API 启动时从引擎 `INFO` 的 `kv_slots` 字段得知路数并开同样
多的引擎连接。每多一路约多占 0.12 GiB 显存（GDN 状态 113 MB 加少量 MTP 流状态）。
MTP 层的 KV/indexer keys 是共享页池里的一层（B0），和主干 KV 一样按页取用，不再每路按
MAX_CONTEXT 预分配（旧版每路多 0.5 GiB（BF16）/ 1 GiB（FP32）；PARALLEL=4 实测省 1.63 GiB）。
rckpt 检查点连同 MTP 层的页一起钉住，恢复后继续 MTP 投机（`tools/b0_verify.sh` 验证）。

## 语义

- **共享上下文**：所有槽位从同一个 KV 页池（256 token 一页）按需取页，类似 llama.cpp
  默认的 unified KV。例如 256K 池可以是 4 条 64K，也可以是 1 条 200K 加 3 条短的。
- **调度**：每个连接一个读线程（`PING/INFO/X` 随时响应），每个 `GEN` 一个工作线程；
  FIFO 票据锁拥有 GPU，持有者在让出点（spec 轮开头、prefill 分块边界、串行 decode
  每个 token）发现有人排队就交出。每条序列算出的结果与它单独运行逐位一致。
- **超出路数**：第 N+1 个请求在引擎（和 API）里排队，直到有槽位空出；排队中可以取消。
- **槽位复用**：新请求优先选 hist 是其严格前缀的空闲槽位（多轮对话免 prefill），其次
  空槽，再次连接已断开的最久未用槽，最后最久未用槽。
- **池不够时**（按顺序）：先淘汰 RAM 检查点；再丢空闲槽位的 KV（最久未用优先）；
  还不够就**中断后来的请求**（在它下一次拿 GPU 时失败）；如果申请者本身就是最后来的，
  则它自己失败。先来的请求永远不会因为后来的请求被牺牲。
- **API 表现**：被中断的请求如果已经流出 token，返回 503
  （`aborted by the engine ... shared KV context full`）；还没出 token 就失败则返回 400。
  池恢复后同样的请求单独重试即可成功。
- kvsnap（SSD 快照）与 rckpt（RAM 检查点）照常工作，检查点同样钉住池里的页。

日志关键字：`[kvpage] pool dry: dropped idle slot`、`aborting the later request`、
`serve: req N failed:`。

## 确定性修复（与并发一起提交）

验证时发现，单 token decode 的 MoE down 投影（`moe()` / `mtp_moe()`）用 float
`atomicAdd` 累加 10 个专家，累加顺序取决于 wave 调度。空闲 GPU 上通常可复现，但时序
一变（另一条序列插在中间、调试同步）就会在最后一位上不同，几千 token 后翻转近似平局
的 token。同一配置单路连跑 3 次就出现过一次在第 2441 个 token 分叉。

现在单 token 路径与小批量 verify 一样走 `k_q4cp_gemv_gd_topk_h16<10>`（专家部分和留在
寄存器里按专家顺序求和，无原子操作）。速度不变；输出与旧二进制在长生成中会有差异
（累加顺序不同），但同一二进制下 unpaged / 分页 / 乱序分页 / 调试同步 / 并发全部逐位一致。
`GDEC_MOE_DOWN_ATOMIC=1` 可切回旧路径做对比。

## 验证

需要先停掉生产服务。一键脚本，最后一行 PASS/FAIL：

```bash
bash build.sh
bash tools/conc_verify.sh
```

三段依次起测试引擎（端口 8732）：

1. **单路**：一组请求（drafter 0/1/3/4、采样、3K 长 prompt、三轮对话）串行跑。
2. **4 路**：同一组请求由 4 个客户端线程并发发送，必须与单路逐位一致（token、结束原因、
   spec 统计）；再测 `INFO kv_slots`、负载下 `PING` 延迟 < 0.5 s、排队中取消、运行中取消、
   同连接重复 `GEN` 被拒、客户端断连释放槽位。
3. **池溢出**：2 路 + `--maxctx 16384`（池只够一条 16K 序列）。A（7K prompt + 3000 token）
   先跑，B 后来：B 在 decode 中被中断，A 与单独运行逐位一致；A2（12K）运行时 B2 连 prefill
   都放不下，B2 直接失败；之后 B2 单独重试成功。

可选：`OLD_BIN=build/gdec.old` 额外对拍旧二进制的单路结果（注意上面的确定性修复会让
旧/新在长输出上合法地不同）；`STAGES="ovf"` 只跑某几段；`CONC_ENV="..."` 附加引擎环境。
