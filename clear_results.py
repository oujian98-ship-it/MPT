"""Clear generated results for selected methods and datasets. Usage: python clear_results.py [all|mpt|METHOD] [set1|set2|set3|set4]. Run without arguments for interactive selection."""

import os
import shutil
import sys


ALL_SETS   = ["set1", "set2", "set3", "set4"]
ALL_METHODS = ["vanilla", "lininterp", "alternating_sampling", "clip_min", "step", "bs", "mpt"]
METHOD_CHOICES = ["all", *ALL_METHODS]
RESULTS_DIR = os.environ.get("RESULTS_DIR", "result")


def clear_method(method_name, target_sets=None):
    """Delete result folders for selected methods in the target sets."""
    if target_sets is None:
        target_sets = ALL_SETS
    methods_to_clear = ALL_METHODS if method_name == "all" else [method_name]

    removed = 0
    skipped = 0

    for set_name in target_sets:
        set_dir = os.path.join(RESULTS_DIR, set_name)
        if not os.path.exists(set_dir):
            print(f"  [SKIP] {set_dir} does not exist")
            continue


        for prompt_id in sorted(os.listdir(set_dir)):
            for method in methods_to_clear:
                method_dir = os.path.join(set_dir, prompt_id, method)
                if os.path.isdir(method_dir):
                    try:
                        shutil.rmtree(method_dir)
                        removed += 1
                        print(f"  [DEL] {method_dir}")
                    except Exception as e:
                        print(f"  [ERR] Failed to delete {method_dir}: {e}")
                else:
                    skipped += 1

    total = removed + skipped
    print(f"\n--- Results ---")
    print(f"  Set(s): {', '.join(target_sets)}")
    print(f"  Method: {method_name}")
    print(f"  Deleted: {removed} folders")
    print(f"  Skipped: {skipped} folders (already absent)")
    print(f"\nYou can now rerun the corresponding run_batch_*.py script.")


def interactive():
    """Prompt the user to select methods and datasets."""
    print("=" * 60)
    print("  Clear generated results — select a method and datasets")
    print("=" * 60)

    print(f"\nAvailable methods:")
    for i, m in enumerate(METHOD_CHOICES):
        print(f"  {i+1}. {m}")

    method_idx = input(f"\nSelect a method (1-{len(METHOD_CHOICES)}): ").strip()
    try:
        method_idx = int(method_idx) - 1
        if not (0 <= method_idx < len(METHOD_CHOICES)):
            print("Invalid selection"); return
    except ValueError:
        print("Please enter a number"); return

    method_name = METHOD_CHOICES[method_idx]

    print(f"\nAvailable sets:")
    for i, s in enumerate(ALL_SETS):
        print(f"  {i+1}. {s}")
    print(f"  0. All")

    set_input = input(f"\nSelect a set (0-{len(ALL_SETS)}): ").strip()
    try:
        set_idx = int(set_input)
        if set_idx == 0:
            target_sets = ALL_SETS
        elif 1 <= set_idx <= len(ALL_SETS):
            target_sets = [ALL_SETS[set_idx - 1]]
        else:
            print("Invalid selection"); return
    except ValueError:
        print("Please enter a number"); return

    print(f"\nAbout to delete: method={method_name}, Set={','.join(target_sets)}")
    confirm = input("Confirm deletion? (y/N): ").strip().lower()
    if confirm == "y":
        clear_method(method_name, target_sets)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:

        method = sys.argv[1]
        if method not in METHOD_CHOICES:
            print(f"Unknown method: {method}")
            print(f"Available: {', '.join(METHOD_CHOICES)}")
            sys.exit(1)

        sets = ALL_SETS
        if len(sys.argv) >= 3:
            sets = [sys.argv[2]]
            if sets[0] not in ALL_SETS:
                print(f"Unknown set: {sets[0]}")
                print(f"Available: {', '.join(ALL_SETS)}")
                sys.exit(1)

        print(f"Deleting results for {method} in {','.join(sets)}...")
        confirm = input("Confirm deletion? (y/N): ").strip().lower()
        if confirm == "y":
            clear_method(method, sets)
        else:
            print("Cancelled.")
    else:

        interactive()
