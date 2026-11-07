"""Entry script of the frozen desktop build (see positioning.spec)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # a frozen Windows process must handle its own child-process start-up
    from breast_seg.apps.desktop import main

    sys.exit(main())
