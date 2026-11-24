"""把 ultralytics 的 YOLO 权重（.pt）转成本项目能读的 .safetensors，
并可选地导出一次前向参考结果，用于 C# 侧的数值对拍（parity）。

用法：

    # 只转换权重（放进程能读到的 data/weights 目录）
    python tools/convert_yolo_weights.py --input yolov8n.pt --output data/weights/yolov8n.safetensors

    # 转换 + 生成前向参考（C# 测试 YoloPretrainedParityTests 会读它）
    python tools/convert_yolo_weights.py --input yolov8n.pt \
        --output data/weights/yolov8n.safetensors \
        --parity data/weights/yolov8n-parity.safetensors --size 320

依赖：torch、safetensors；转换 ultralytics 的 .pt 需要装 ultralytics（pickle 里有它的类）。

注意：权重目录是 **exe 同级 data/weights**（例如
src/Label2Train.App/bin/Debug/net10.0/data/weights），或者项目根的 data/weights——
程序会依次在两个位置找。
"""

from __future__ import annotations

import argparse
import sys


def load_checkpoint(path: str):
    import torch

    # 纯 state_dict 的检查点可以 weights_only=True；ultralytics 的 .pt 必须 False
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_model(checkpoint):
    """从 ultralytics 检查点里取出模型（结构 + 权重）。"""
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if model is None:
        model = checkpoint
    if not hasattr(model, "state_dict"):
        raise SystemExit(
            "输入里没有模型对象，只有张量。请直接给 ultralytics 的 .pt（含 'model' 键）。"
        )
    return model


def save_weights(model, output: str) -> int:
    from safetensors.torch import save_file

    tensors = {}
    for name, value in model.state_dict().items():
        if not value.is_floating_point():
            # num_batches_tracked 等整型 buffer：本项目 load_state_dict 不需要
            continue
        tensors[name] = value.detach().float().cpu().contiguous()

    save_file(tensors, output)
    return len(tensors)


def save_parity(model, output: str, size: int, seed: int) -> None:
    import torch
    from safetensors.torch import save_file

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, size, size, generator=generator, dtype=torch.float32)

    head = model.model[-1]
    # 有没有 one-to-one 分支看模块，不看 head.end2end：ultralytics 8.4 的 end2end
    # 是个属性（_end2end 或 cv2 被融合掉才为真），检查点里保存的双头模型通常是 False，
    # 但 one2one_cv2/one2one_cv3 确实存在、训练与导出都要用。
    end2end = getattr(head, "one2one_cv2", None) is not None

    tensors = {"input": x.contiguous()}

    if not end2end:
        with torch.no_grad():
            y = model(x)

        # eval 下 DetectionModel.forward 返回 (解码后的 [B,4+nc,N], 原始预测 dict)
        if not (isinstance(y, (tuple, list)) and len(y) == 2):
            raise SystemExit("模型没有返回 (decoded, raw) 二元组，无法导出参考")

        decoded, raw = y[0], y[1]
        if isinstance(raw, dict) and "one2many" in raw:
            raw = raw["one2many"]

        # ultralytics 默认输出 xywh（dist2bbox 的 xywh=True 分支）；本项目解码输出 xyxy。
        # 参考数据统一存成 xyxy，C# 侧才能直接逐元素比。
        if not getattr(head, "xyxy", False):
            cx, cy, w, h = decoded[:, 0], decoded[:, 1], decoded[:, 2], decoded[:, 3]
            decoded = torch.cat(
                [torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), decoded[:, 4:]],
                1,
            )

        tensors["output"] = decoded.float().contiguous()

        # 原始分支拆开存，便于 C# 侧先比「网络前向」再比「解码公式」
        if isinstance(raw, dict) and "boxes" in raw:
            tensors["boxes"] = raw["boxes"].float().contiguous()
            tensors["scores"] = raw["scores"].float().contiguous()
    else:
        # YOLO26：eval 下 forward 只走 one-to-many 并直接做主处理（[B,max_det,6]），
        # 拿不到两套分支的原始输出，所以 hook 一次检测头输入，手工调 forward_head
        captured = {}

        def hook(_module, args):
            captured.setdefault("feats", args[0])

        handle = head.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                y = model(x)
        finally:
            handle.remove()

        # 双头检查点若仍走 one-to-many 推理，model(x) 会返回 (decoded, raw)。
        # 把它按 xyxy 存下来，C# 侧才能顺带对拍解码公式（reg_max = 1 时没有 DFL）。
        if isinstance(y, (tuple, list)) and len(y) == 2 and y[0].dim() == 3:
            decoded = y[0]
            if not getattr(head, "xyxy", False):
                cx, cy, w, h = decoded[:, 0], decoded[:, 1], decoded[:, 2], decoded[:, 3]
                decoded = torch.cat(
                    [torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), decoded[:, 4:]],
                    1,
                )
            tensors["output"] = decoded.float().contiguous()

        feats = captured.get("feats")
        if feats is None:
            raise SystemExit("没能捕获检测头输入，无法导出 YOLO26 参考")

        with torch.no_grad():
            one2many = head.forward_head(feats, head.cv2, head.cv3)
            # 官方训练时 one-to-one 分支吃 detach 后的特征，数值与不 detach 相同
            one2one = head.forward_head(
                [item.detach() for item in feats], head.one2one_cv2, head.one2one_cv3
            )

        for prefix, branch in (("", one2many), ("1", one2one)):
            tensors[f"boxes{prefix}"] = branch["boxes"].float().contiguous()
            tensors[f"scores{prefix}"] = branch["scores"].float().contiguous()

    save_file(tensors, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ultralytics 的 .pt/.pth")
    parser.add_argument("--output", required=True, help="输出 .safetensors")
    parser.add_argument("--parity", default=None, help="可选：输出前向参考 .safetensors")
    parser.add_argument("--size", type=int, default=320, help="参考前向的输入边长（32 的倍数）")
    parser.add_argument("--seed", type=int, default=20261120, help="参考输入的随机种子")
    args = parser.parse_args()

    if args.size % 32 != 0:
        raise SystemExit("--size 必须是 32 的倍数")

    checkpoint = load_checkpoint(args.input)
    model = extract_model(checkpoint)

    if hasattr(model, "float"):
        model = model.float()
    if hasattr(model, "eval"):
        model.eval()

    count = save_weights(model, args.output)
    print(f"权重已写入 {args.output}（{count} 个张量）")

    if args.parity:
        save_parity(model, args.parity, args.size, args.seed)
        print(f"参考前向已写入 {args.parity}（输入 {args.size}x{args.size}，seed {args.seed}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
