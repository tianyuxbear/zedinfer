import torch

import zedinfer
from zedinfer.test_utils import benchmark, check_equal, random_tensor


def test_op_linear(
    out_shape,
    x_shape,
    w_shape,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   out {out_shape}, x {x_shape}, w {w_shape}, bias {use_bias}, dtype <{dtype_name}>"
    )
    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.01)
    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)

    bias, bias_ = None, None
    if use_bias:
        bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)

    out, out_ = random_tensor(out_shape, dtype_name, device_name)
    out = torch.nn.functional.linear(x, w, bias)
    zedinfer.Ops.linear(out_, x_, w_, bias_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch.nn.functional.linear(x, w, bias),
            lambda: zedinfer.Ops.linear(out_, x_, w_, bias_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [
        ((1, 1536), (1, 1536), (1536, 1536), True),
        ((1, 8960), (1, 1536), (8960, 1536), True),
        ((1, 1536), (1, 8960), (1536, 8960), True),
        ((128, 1536), (128, 1536), (1536, 1536), True),
        ((128, 8960), (128, 1536), (8960, 1536), True),
        ((128, 1536), (128, 8960), (1536, 8960), True),
        ((1, 4096), (1, 4096), (4096, 4096), True),
        ((1, 12288), (1, 4096), (12288, 4096), True),
        ((1, 4096), (1, 12288), (4096, 12288), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
        ((512, 12288), (512, 4096), (12288, 4096), True),
        ((512, 4096), (512, 12288), (4096, 12288), True),
        ((1, 5120), (1, 5120), (5120, 5120), True),
        ((1, 27648), (1, 5120), (27648, 5120), True),
        ((1, 5120), (1, 27648), (5120, 27648), True),
        ((1024, 5120), (1024, 5120), (5120, 5120), True),
        ((1024, 27648), (1024, 5120), (27648, 5120), True),
        ((1024, 5120), (1024, 27648), (5120, 27648), True),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 5e-5, 5e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 5e-2, 5e-2),
    ]
    print(f"Testing Ops.linear on {args.device}")
    for shapes in testShapes:
        out_shape, x_shape, w_shape, use_bias = shapes
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_linear(
                out_shape=out_shape,
                x_shape=x_shape,
                w_shape=w_shape,
                use_bias=use_bias,
                dtype_name=dtype_name,
                atol=atol,
                rtol=rtol,
                device_name=args.device,
                profile=args.profile,
            )

    print("\033[92mTest passed!\033[0m\n")
