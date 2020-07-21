"""Integration test based on particle_in_field turbopy app

This integration test simply runs the particle_in_field app and compares the output files
to "good" output.
"""
import io

def test_pif(pif_run):
    for filename in ['e_0.5', 'grid', 'particle_p', 'particle_x', 'time']:
        assert (list(io.open(f'tests/fixtures/particle_in_field/output/{filename}.csv')) ==
                list(io.open(f'tmp/particle_in_field/output/{filename}.csv')))
