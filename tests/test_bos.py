"""Integration test based on block_on_spring turbopy app

This integration test simply runs the block_on_spring app and compares the output files
to "good" output.
"""
import numpy as np
import pytest
from turbopy import Simulation

def test_bos_forwardeuler(bos_run, tmp_path):
    """Tests block_on_spring app with ForwardEuler ComputeTool and compares to
    output files with a "good" output.
    """

    bos_run["PhysicsModules"]["BlockOnSpring"]["pusher"] = "BlockForwardEuler"
    bos_run["Diagnostics"]["directory"] = tmp_path / "output"
    sim = Simulation(bos_run)
    sim.run()

    for filename in ['block_p', 'block_x', 'time']:
        ref_data = np.genfromtxt(f'tests/fixtures/block_on_spring/output_forwardeuler/{filename}.csv',
                                 delimiter=',')
        tmp_data = np.genfromtxt(tmp_path / f'output/{filename}.csv',
                                 delimiter=',')
        assert np.allclose(ref_data, tmp_data, rtol=1e-05, atol=1e-08)


def test_bos_backwardeuler(bos_run, tmp_path):
    """Tests block_on_spring app with BackwardEuler ComputeTool and compares to
    output files with a "good" output.
    """

    bos_run["PhysicsModules"]["BlockOnSpring"]["pusher"] = "BackwardEuler"
    bos_run["Diagnostics"]["directory"] = tmp_path / "output"
    sim = Simulation(bos_run)
    sim.run()

    for filename in ['block_p', 'block_x', 'time']:
        ref_data = np.genfromtxt(f'tests/fixtures/block_on_spring/output_backwardeuler/{filename}.csv',
                                 delimiter=',')
        tmp_data = np.genfromtxt(tmp_path / f'output/{filename}.csv',
                                 delimiter=',')
        assert np.allclose(ref_data, tmp_data, rtol=1e-05, atol=1e-08)


def test_bos_leapfrog(bos_run, tmp_path):
    """Tests block_on_spring app with LeapFrog ComputeTool and compares to
    output files with a "good" output.
    """

    bos_run["PhysicsModules"]["BlockOnSpring"]["pusher"] = "Leapfrog"
    bos_run["Diagnostics"]["directory"] = tmp_path / "output"
    sim = Simulation(bos_run)
    sim.run()

    for filename in ['block_p', 'block_x', 'time']:
        ref_data = np.genfromtxt(f'tests/fixtures/block_on_spring/output_leapfrog/{filename}.csv',
                                 delimiter=',')
        tmp_data = np.genfromtxt(tmp_path / f'output/{filename}.csv',
                                 delimiter=',')
        assert np.allclose(ref_data, tmp_data, rtol=1e-05, atol=1e-08)

