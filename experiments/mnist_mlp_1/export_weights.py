import struct
import torch

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_FILE = SCRIPT_DIR / "mnist_mlp.pth"
OUTPUT_FILE = SCRIPT_DIR / "mnist_mlp_weights.bin"


def write_string(file, value):
    encoded = value.encode("utf-8")

    file.write(struct.pack("<I", len(encoded)))
    file.write(encoded)


def write_tensor(file, name, tensor):
    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    values = tensor.flatten().tolist()

    write_string(file, name)

    file.write(struct.pack("<I", tensor.ndim))

    for dimension in tensor.shape:
        file.write(struct.pack("<Q", dimension))

    file.write(struct.pack("<Q", len(values)))

    for value in values:
        file.write(struct.pack("<f", value))


def main():
    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    print(f"Loading: {MODEL_FILE}")

    with open(OUTPUT_FILE, "wb") as file:
        tensors = list(checkpoint.items())

        file.write(struct.pack("<I", len(tensors)))

        for name, tensor in tensors:
            write_tensor(file, name, tensor)

            print(
                f"{name:12s} "
                f"shape={tuple(tensor.shape)} "
                f"values={tensor.numel()}"
            )

    print()
    print(f"Exported to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()