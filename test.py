import time
import psutil
import os
import scipy.sparse as sp

from src.ml_jnmf_new import ML_JNMF as ML_New     # 新版本
from src.ml_jnmf import ML_JNMF as ML_Old            # 老版本
from main import preprocess      # 数据接口一致

# ================= 工具函数 =================

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_model(model, name, r=50, iters=3):
    print(f"\n================= 测试模型: {name} =================")
    print(f"初始内存: {get_memory_usage():.2f} MB")

    # --- 初始化 ---
    print("\n--- matrixInit ---")
    mem_before = get_memory_usage()
    start = time.time()
    model.matrixInit(r=r)
    t_init = time.time() - start
    mem_after = get_memory_usage()

    print(f"初始化耗时: {t_init:.4f} s")
    print(f"初始化内存: {mem_before:.2f} → {mem_after:.2f} MB")

    # --- 迭代 ---
    iter_times = []
    losses = []

    for i in range(iters):
        print(f"\n--- 第 {i+1} 次迭代 ---")
        
        mem_before = get_memory_usage()
        start = time.time()
        
        model.multiplicativeUpdate()
        loss = model.calculateLoss()

        t_iter = time.time() - start
        mem_after = get_memory_usage()

        iter_times.append(t_iter)
        losses.append(loss)

        print(f"迭代耗时: {t_iter:.4f} s")
        print(f"迭代内存: {mem_before:.2f} → {mem_after:.2f} MB")
        print(f"loss = {loss:.6f}")

    return {
        "name": name,
        "init_time": t_init,
        "iter_times": iter_times,
        "losses": losses,
    }


# ================== 主流程 ==================

print("加载 Cora 数据...")
la, ls, li = preprocess("cora",preprocessParams={"order":5,"decay":2})

# 新旧模型
new_model = ML_New(la.copy(), ls.copy(), li.copy())
old_model = ML_Old(la.copy(), ls.copy(), li.copy())

print("\n===== 开始性能对比 =====")

result_old = benchmark_model(old_model, "旧版本", r=50, iters=3)
result_new = benchmark_model(new_model, "新版本", r=50, iters=3)

# ================== 总结报告 ==================

print("\n\n================ 性能对比报告 ================")

print(f"\n初始化耗时:    旧 {result_old['init_time']:.4f} s    |  新 {result_new['init_time']:.4f} s")
print(f"迭代平均耗时:  旧 {sum(result_old['iter_times'])/3:.4f} s /iter  |  新 {sum(result_new['iter_times'])/3:.4f} s /iter")

print("\n详细迭代耗时：")
for i in range(3):
    print(f"第{i+1}次迭代: 旧 {result_old['iter_times'][i]:.4f} s | 新 {result_new['iter_times'][i]:.4f} s")

print("\nloss 对比：")
for i in range(3):
    print(f"第{i+1}轮: 旧 {result_old['losses'][i]:.6f} | 新 {result_new['losses'][i]:.6f}")

print("\n=====================================================")
print("对比结束。")
