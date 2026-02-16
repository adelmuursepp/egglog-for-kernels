## Data Movement Calculations

Every operation that moves data between memory levels has a cost in bytes. The cost is `rows * cols * dtype_bytes * loop_iters`.

### Naive Path

| Operation | What it does | Bytes per iter | Iters | Total |
|-----------|-------------|----------------|-------|-------|
| LDS(Q) | global -> shared | 16,384 | 1 | 16,384 |
| LDS(K) | global -> shared | 16,384 | 8 | 131,072 |
| LDS(V) | global -> shared | 16,384 | 8 | 131,072 |
| WGMMA(Q_smem, K_smem) | implicit smem->reg for both operands | 16,384 + 16,384 = 32,768 | 8 | 262,144 |
| Elementwise(QK) | pure compute | 0 | 8 | 0 |
| WGMMA(A, V_smem) | implicit smem->reg for V (A is in registers) | 16,384 | 8 | 131,072 |
| STS(output) | registers -> shared | 32,768 | 8 | 262,144 |
| STG(output) | shared -> global | 32,768 | 8 | 262,144 |
| | | | **Total** | **1,196,032** |

The first WGMMA costs 32,768 bytes per iteration because both Q and K are in shared memory, so the hardware implicitly loads both to registers. Note the output tile is 128x64 in fp32 = 32,768 bytes.

### Rearranged Path (Optimal)

| Operation | What it does | Bytes per iter | Iters | Total |
|-----------|-------------|----------------|-------|-------|
| LDS(Q) | global -> shared | 16,384 | 1 | 16,384 |
| LDR(Q_smem) | shared -> registers | 16,384 | 1 | 16,384 |
| Elementwise(Q_reg) | pure compute | 0 | 1 | 0 |
| LDS(K) | global -> shared | 16,384 | 8 | 131,072 |
| WGMMA(Q_scaled, K_smem) | implicit smem->reg for K only (Q is in registers) | 16,384 | 8 | 131,072 |
| LDS(V) | global -> shared | 16,384 | 8 | 131,072 |
| WGMMA(QK, V_smem) | implicit smem->reg for V (QK is in registers) | 16,384 | 8 | 131,072 |
| STS(output) | registers -> shared | 32,768 | 8 | 262,144 |
| STG(output) | shared -> global | 32,768 | 8 | 262,144 |
| | | | **Total** | **1,081,344** |

The savings come from the first WGMMA: Q is already in registers, so only K needs the implicit load. The explicit LDR costs 16,384 bytes once, replacing the implicit 16,384-byte load of Q that happened 8 times (131,072 bytes).

**Savings: 1,196,032 - 1,081,344 = 114,688 bytes (saving 131,072, spending 16,384)**