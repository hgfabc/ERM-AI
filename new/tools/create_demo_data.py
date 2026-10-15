"""Create a small, explicitly synthetic eight-stream cohort for smoke tests."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

STREAMS = ["_vertical", "_horizontal", "color_full", "color_inner",
           "color_outer", "gray_full", "gray_inner", "gray_outer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patients", type=int, default=24)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.patients < 12 or args.patients % 2:
        parser.error("--patients must be an even number >= 12 for the two-class nested smoke split.")
    if root.exists() and any(root.iterdir()):
        parser.error("Output directory is not empty; choose a new destination.")
    rng = np.random.default_rng(42)
    for index in range(args.patients):
        label = "bad" if index % 2 == 0 else "good"
        for eye in ("OD", "OS"):
            case = f"DEMO{index:04d}_EXAM0_{eye}"
            folder = root / label / case
            folder.mkdir(parents=True, exist_ok=True)
            image = rng.integers(0, 100, (48, 48, 3), dtype=np.uint8)
            image[:, :24, index % 2] += 100
            for stream_index, stream in enumerate(STREAMS):
                Image.fromarray(np.roll(image, stream_index, axis=1)).save(folder / f"{case}_{stream.lstrip('_')}.png")
    print(f"Created {args.patients * 2} SYNTHETIC cases in {root}")


if __name__ == "__main__":
    main()
