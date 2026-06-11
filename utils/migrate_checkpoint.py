"""
迁移脚本：将方案1修复前的旧checkpoint补全为兼容新模型结构的checkpoint。

背景：
  修复前 build_transformer_local 在 ArcFace/Cosface/AMSoftmax/Circle 模式下
  只创建了 self.classifier（全局分支），缺少局部分支的 classifier_1~4。
  修复后新增了 4 个 classifier，旧 checkpoint 加载会因缺少 key 而报错。

策略：
  从已训练好的全局 classifier.weight 复制到缺失的 classifier_1~4.weight。
  全局分支的 ArcFace 分类器已经学到了有意义的类中心，以此为起点让局部分支
  快速适应局部特征，远优于随机初始化。

用法：
  python utils/migrate_checkpoint.py \
      --input logs/BallShow_vit_transreid_stride/transformer_120.pth \
      --output logs/BallShow_vit_transreid_stride/transformer_120_migrated.pth

  # 批量迁移一个目录下所有 .pth 文件
  python utils/migrate_checkpoint.py \
      --input-dir logs/BallShow_vit_transreid_stride/ \
      --output-dir logs/BallShow_vit_transreid_stride_migrated/
"""

import os
import sys
import argparse
import torch
from collections import OrderedDict


# 需要补全的 key 映射：{缺失的key: 用于复制的源key}
# 支持 DDP prefix（module.）和普通 prefix
MISSING_KEY_TEMPLATES = [
    "classifier_{i}.weight",
    "module.classifier_{i}.weight",
]


def detect_prefix(param_dict):
    """检测 state_dict 使用的 key 前缀。"""
    keys = list(param_dict.keys())
    has_module = any(k.startswith("module.") for k in keys)
    has_classifier = any("classifier.weight" in k for k in keys)
    has_classifier_1 = any("classifier_1.weight" in k for k in keys)
    return {
        "has_module": has_module,
        "has_classifier": has_classifier,
        "has_classifier_1": has_classifier_1,
        "total_keys": len(keys),
    }


def migrate_checkpoint(state_dict, dry_run=False):
    """
    补全缺失的 classifier_1~4 权重。

    Args:
        state_dict: 原始 OrderedDict
        dry_run: 仅打印信息不实际修改

    Returns:
        (new_state_dict, report) — 补全后的 state_dict 和变更报告
    """
    prefix = "module." if detect_prefix(state_dict)["has_module"] else ""
    src_key = f"{prefix}classifier.weight"
    report = {"missing": [], "created": [], "already_exists": []}

    if src_key not in state_dict:
        report["missing"].append(src_key)
        return state_dict, report

    for i in range(1, 5):  # classifier_1 ~ classifier_4
        dst_key = f"{prefix}classifier_{i}.weight"
        if dst_key in state_dict:
            report["already_exists"].append(dst_key)
            continue

        src_tensor = state_dict[src_key]

        # 验证形状一致性（classifier 和 classifier_i 都是 [num_classes, feat_dim]）
        if src_tensor.dim() != 2:
            print(f"  ⚠ 警告: {src_key} 形状异常 {src_tensor.shape}，跳过 {dst_key}")
            report["missing"].append(dst_key)
            continue

        report["created"].append(dst_key)
        if not dry_run:
            state_dict[dst_key] = src_tensor.clone()

    return state_dict, report


def print_report(input_path, info_before, report, output_path=None):
    """打印迁移报告。"""
    print(f"\n{'='*60}")
    print(f"文件: {input_path}")
    print(f"  Key 总数: {info_before['total_keys']}")
    print(f"  DDP prefix (module.): {info_before['has_module']}")
    print(f"  已有 classifier.weight: {info_before['has_classifier']}")
    print(f"  已有 classifier_1.weight: {info_before['has_classifier_1']}")

    if report["missing"] and "classifier.weight" not in report["missing"]:
        print(f"\n  🔴 无法补全的 key ({len(report['missing'])}个):")
        for k in report["missing"]:
            print(f"     - {k}")

    if not report["created"] and not report["already_exists"]:
        if info_before["has_classifier_1"]:
            print(f"\n  ✅ 无需迁移 — checkpoint 已兼容新模型")
        elif not info_before["has_classifier"]:
            print(f"\n  ⚠ 跳过 — 未找到 classifier.weight（可能不是 JPM + ArcFace 的 checkpoint）")
        return

    if report["already_exists"]:
        print(f"\n  ✅ 已存在 ({len(report['already_exists'])}个):")
        for k in report["already_exists"]:
            print(f"     - {k}")

    if report["created"]:
        print(f"\n  🟢 新建 ({len(report['created'])}个):")
        for k in report["created"]:
            print(f"     - {k}")
        print(f"\n     来源: classifier.weight → classifier_{{1..4}}.weight")
        print(f"     形状: {report['created']} 共用同一 [num_classes, 768] 权重矩阵")

    if output_path:
        print(f"\n  输出: {output_path}")
    print(f"{'='*60}")


def process_single_file(input_path, output_path, dry_run=False):
    """处理单个 checkpoint 文件。"""
    print(f"\n处理: {input_path}")

    if not os.path.exists(input_path):
        print(f"  ❌ 文件不存在: {input_path}")
        return False

    checkpoint = torch.load(input_path, map_location="cpu")

    # 处理不同的 checkpoint 格式
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        is_wrapped = True
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        is_wrapped = True
    else:
        state_dict = checkpoint
        is_wrapped = False

    if not isinstance(state_dict, OrderedDict):
        state_dict = OrderedDict(state_dict)

    info_before = detect_prefix(state_dict)
    new_state_dict, report = migrate_checkpoint(state_dict, dry_run=dry_run)

    print_report(input_path, info_before, report, output_path if not dry_run else None)

    if not report["created"]:
        return False  # 无需保存

    if dry_run:
        return True

    # 恢复原始包装格式并保存
    if is_wrapped:
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = new_state_dict
        elif "model" in checkpoint:
            checkpoint["model"] = new_state_dict
        output_obj = checkpoint
    else:
        output_obj = new_state_dict

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(output_obj, output_path)
    print(f"  ✅ 已保存: {output_path}")
    return True


def process_directory(input_dir, output_dir, dry_run=False):
    """批量处理目录下所有 .pth 文件。"""
    if not os.path.isdir(input_dir):
        print(f"❌ 目录不存在: {input_dir}")
        return

    pth_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".pth") and os.path.isfile(os.path.join(input_dir, f))
    ])

    if not pth_files:
        print(f"❌ 目录中无 .pth 文件: {input_dir}")
        return

    print(f"批量迁移 {len(pth_files)} 个文件:")
    for f in pth_files:
        print(f"  - {f}")

    migrated_count = 0
    for f in pth_files:
        input_path = os.path.join(input_dir, f)
        name, ext = os.path.splitext(f)
        output_path = os.path.join(output_dir, f"{name}{ext}")
        if process_single_file(input_path, output_path, dry_run=dry_run):
            migrated_count += 1

    print(f"\n总计: {len(pth_files)} 个文件, {migrated_count} 个需要迁移")


def main():
    parser = argparse.ArgumentParser(
        description="迁移旧 ReID checkpoint 以兼容方案1修复后的模型结构"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="单个旧 checkpoint 路径"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出路径（单文件模式）"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="批量模式：输入目录（处理所有 .pth 文件）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="批量模式：输出目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅检查不实际写入"
    )

    args = parser.parse_args()

    if args.input_dir:
        output_dir = args.output_dir or args.input_dir + "_migrated"
        process_directory(args.input_dir, output_dir, dry_run=args.dry_run)
    elif args.input:
        output_path = args.output or args.input.replace(".pth", "_migrated.pth")
        process_single_file(args.input, output_path, dry_run=args.dry_run)
    else:
        parser.print_help()
        print("\n示例:")
        print("  单文件: python utils/migrate_checkpoint.py --input old.pth --output new.pth")
        print("  批量:   python utils/migrate_checkpoint.py --input-dir logs/old/ --output-dir logs/new/")
        print("  预览:   python utils/migrate_checkpoint.py --input old.pth --dry-run")


if __name__ == "__main__":
    main()
