"""Compatibility entry point for the relocated clean-layout compiler."""

from clean_layout_data_engine.orientation_intent_compiler import *  # noqa: F401,F403
from clean_layout_data_engine.orientation_intent_compiler import main


if __name__ == "__main__":
    main()
