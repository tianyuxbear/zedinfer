from zedinfer.test_utils import benchmark, check_equal, random_int_tensor, random_tensor

import zedinfer


def torch_embedding(out, idx, embd):
    out[:] = embd[idx]


def test_op_embedding(
    idx_shape,
    embd_shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   idx_shape {idx_shape} embd_shape {embd_shape} dtype <{dtype_name}>")

    embd, embd_ = random_tensor(embd_shape, dtype_name, device_name)
    idx, idx_ = random_int_tensor(idx_shape, device_name, high=embd_shape[0])

    out_shape = (idx_shape[0], embd_shape[1])
    out, out_ = random_tensor(out_shape, dtype_name, device_name)

    torch_embedding(out, idx, embd)
    zedinfer.Ops.embedding(out_, idx_, embd_)

    assert check_equal(out_, out, strict=True)

    if profile:
        benchmark(
            lambda: torch_embedding(out, idx, embd),
            lambda: zedinfer.Ops.embedding(out_, idx_, embd_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    testShapes = [
        # (idx_shape, embd_shape)
        ((1,), (151936, 1536)),
        ((128,), (151936, 1536)),
        ((1,), (151936, 4096)),
        ((512,), (151936, 4096)),
        ((1,), (152064, 5120)),
        ((1024,), (152064, 5120)),
    ]

    testDtype = [
        "f32",
        "f16",
        "bf16",
    ]

    print(f"Testing Ops.embedding on {args.device}")
    for idx_shape, embd_shape in testShapes:
        for dtype_name in testDtype:
            test_op_embedding(
                idx_shape, embd_shape, dtype_name, args.device, args.profile
            )
    print("\033[92mTest passed!\033[0m\n")
