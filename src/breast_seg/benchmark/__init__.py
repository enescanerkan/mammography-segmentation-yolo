"""Benchmarks of the released pipelines.

``breast_seg.benchmark.runtime``: end-to-end runtime of one MLO-CC pair, from the two DICOM files to the adequacy
decision, for every pipeline in one interleaved session (``breast-seg benchmark runtime``); its summary and the
disturbed-pass rule are in ``breast_seg.benchmark.summary`` (``breast-seg benchmark summary``).

Nothing heavy is imported here: :mod:`breast_seg.benchmark.protocol` must be able to pin the process to its CPU
cores before numpy, OpenCV or torch start their thread pools.
"""
