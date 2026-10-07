"""
clear_results.py
---------------
删除指定方法或全部方法的生成结果，让 batch 脚本重新跑。

用法:
  python clear_results.py              # 交互选择方法 + set
  python clear_results.py all          # 清除所有方法、所有 set
  python clear_results.py mpt          # 直接指定方法（清除所有 set）
  python clear_results.py mpt set4     # 指定方法 + 指定 set
  python clear_results.py all set4     # 清除 set4 的全部方法

支持的方法: all, vanilla, lininterp, alternating_sampling, clip_min, step, bs, mpt
"""

import os
import shutil
import sys


ALL_SETS   = ["set1", "set2", "set3", "set4"]
ALL_METHODS = ["vanilla", "lininterp", "alternating_sampling", "clip_min", "step", "bs", "mpt"]
METHOD_CHOICES = ["all", *ALL_METHODS]
RESULTS_DIR = os.environ.get("RESULTS_DIR", "result")


def clear_method(method_name, target_sets=None):
    """删除指定方法（或 all）在目标 set 中的所有结果文件夹。"""
    if target_sets is None:
        target_sets = ALL_SETS
    methods_to_clear = ALL_METHODS if method_name == "all" else [method_name]

    removed = 0
    skipped = 0

    for set_name in target_sets:
        set_dir = os.path.join(RESULTS_DIR, set_name)
        if not os.path.exists(set_dir):
            print(f"  [SKIP] {set_dir} 不存在")
            continue

        # 遍历该 set 下所有 prompt 文件夹和目标方法目录。
        for prompt_id in sorted(os.listdir(set_dir)):
            for method in methods_to_clear:
                method_dir = os.path.join(set_dir, prompt_id, method)
                if os.path.isdir(method_dir):
                    try:
                        shutil.rmtree(method_dir)
                        removed += 1
                        print(f"  [DEL] {method_dir}")
                    except Exception as e:
                        print(f"  [ERR] 删除失败 {method_dir}: {e}")
                else:
                    skipped += 1

    total = removed + skipped
    print(f"\n--- 结果 ---")
    print(f"  Set(s): {', '.join(target_sets)}")
    print(f"  Method: {method_name}")
    print(f"  已删除: {removed} 个文件夹")
    print(f"  无需删: {skipped} 个（本来就没有）")
    print(f"\n现在可以重新运行对应的 run_batch_*.py 了。")


def interactive():
    """交互模式：让用户选择方法和 set"""
    print("=" * 60)
    print("  清理生成结果 — 选择要删除的方法和数据集")
    print("=" * 60)

    print(f"\n可用方法:")
    for i, m in enumerate(METHOD_CHOICES):
        print(f"  {i+1}. {m}")

    method_idx = input(f"\n选择方法 (1-{len(METHOD_CHOICES)}): ").strip()
    try:
        method_idx = int(method_idx) - 1
        if not (0 <= method_idx < len(METHOD_CHOICES)):
            print("无效选择"); return
    except ValueError:
        print("请输入数字"); return

    method_name = METHOD_CHOICES[method_idx]

    print(f"\n可选 Set:")
    for i, s in enumerate(ALL_SETS):
        print(f"  {i+1}. {s}")
    print(f"  0. 全部")

    set_input = input(f"\n选择 Set (0-{len(ALL_SETS)}): ").strip()
    try:
        set_idx = int(set_input)
        if set_idx == 0:
            target_sets = ALL_SETS
        elif 1 <= set_idx <= len(ALL_SETS):
            target_sets = [ALL_SETS[set_idx - 1]]
        else:
            print("无效选择"); return
    except ValueError:
        print("请输入数字"); return

    print(f"\n即将删除: 方法={method_name}, Set={','.join(target_sets)}")
    confirm = input("确认删除? (y/N): ").strip().lower()
    if confirm == "y":
        clear_method(method_name, target_sets)
    else:
        print("已取消。")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # 命令行模式
        method = sys.argv[1]
        if method not in METHOD_CHOICES:
            print(f"未知方法: {method}")
            print(f"可用: {', '.join(METHOD_CHOICES)}")
            sys.exit(1)

        sets = ALL_SETS
        if len(sys.argv) >= 3:
            sets = [sys.argv[2]]
            if sets[0] not in ALL_SETS:
                print(f"未知 set: {sets[0]}")
                print(f"可用: {', '.join(ALL_SETS)}")
                sys.exit(1)

        print(f"删除 {method} 在 {','.join(sets)} 中的结果...")
        confirm = input("确认删除? (y/N): ").strip().lower()
        if confirm == "y":
            clear_method(method, sets)
        else:
            print("已取消。")
    else:
        # 交互模式
        interactive()
