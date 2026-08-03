from __future__ import annotations

import argparse
from pathlib import Path


PLACEHOLDER = "__PREDICTION_DATA__"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone prediction gallery")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("HTML template must contain exactly one data placeholder")
    data = args.data.read_text(encoding="utf-8").replace("</script", "<\\/script")
    output = template.replace(PLACEHOLDER, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
