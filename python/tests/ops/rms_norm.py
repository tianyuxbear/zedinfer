import torch
from zedinfer.test_utils import benchmark, check_equal, random_tensor

import zedinfer


def torch_rms_norm(out, x, weight, eps):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    out.copy_(weight * x / rms)


def test_op_rms_norm(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name)
    w, w_ = random_tensor((shape[1],), dtype_name, device_name)
    eps = 1e-5

    c, c_ = random_tensor(shape, dtype_name, device_name)
    torch_rms_norm(c, x, w, eps)
    zedinfer.Ops.rms_norm(c_, x_, w_, eps)

    assert check_equal(c_, c, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_rms_norm(c, x, w, eps),
            lambda: zedinfer.Ops.rms_norm(c_, x_, w_, eps),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        (1, 1536),
        (128, 1536),
        (1, 4096),
        (512, 4096),
        (1, 5120),
        (1024, 5120),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 5e-3, 5e-3),
        ("bf16", 5e-2, 5e-2),
    ]
    print(f"Testing Ops.rms_norm on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_rms_norm(shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
