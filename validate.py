"""Repository root validation wrapper for the OpenEnv submission."""

from __future__ import annotations

from openenv.validate import validate_environment


def main() -> int:
    results = validate_environment()
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
