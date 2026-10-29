"""Publication output for B, using the same statistics and graphics as A."""
from paper_a_report import export_paper_a


def export_paper_b(run_directory, png_dpi=1200, allow_pilot=False):
    return export_paper_a(run_directory, png_dpi, allow_pilot, experiment='B')
