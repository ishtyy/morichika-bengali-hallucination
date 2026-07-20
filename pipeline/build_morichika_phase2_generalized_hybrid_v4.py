"""Canonical entry point for the full-coverage MORICHIKA generalized v4 kernel.

The implementation remains in the v3-derived transformation module so the
closed-book code path stays byte-identical to its audited predecessor.  This
entry point prevents operators from confusing the new v4 output/kernel slug
with the previously built v3 artifact.
"""

from pipeline.build_morichika_phase2_generalized_hybrid_v3 import (  # noqa: F401
    KERNEL_ID,
    NOTEBOOK_NAME,
    OUT,
    build,
    main,
    transformed_runner,
)


if __name__ == "__main__":
    main()
