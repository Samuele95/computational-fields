"""Tests for network topology."""

from __future__ import annotations

import numpy as np

from computational_fields.simulation.network import Network


class TestNetwork:
    def test_add_device(self):
        net = Network(comm_range=2.0)
        d = net.add_device((0, 0), {"temp": 20})
        assert d.id == 0
        assert d.sensors["temp"] == 20

    def test_neighbor_discovery(self):
        net = Network(comm_range=1.5)
        net.add_device((0, 0), device_id=0)
        net.add_device((1, 0), device_id=1)
        net.add_device((3, 0), device_id=2)  # too far from 0
        net.update_neighbors()
        assert 1 in net.devices[0].neighbors
        assert 2 not in net.devices[0].neighbors
        assert 0 in net.devices[1].neighbors
        assert 2 not in net.devices[1].neighbors

    def test_grid_factory(self):
        net = Network.grid(3, 3, spacing=1.0)
        assert len(net.devices) == 9
        # With comm_range=1.5 (default), diagonals (~1.41) are included.
        # Corner device has 3 neighbors (right, above, diagonal).
        d0 = net.devices[0]
        assert len(d0.neighbors) == 3

    def test_grid_center_has_eight_neighbors(self):
        net = Network.grid(3, 3, spacing=1.0)
        # Center device (1,1) = id 4, with diagonals included has 8 neighbors
        d4 = net.devices[4]
        assert len(d4.neighbors) == 8

    def test_grid_no_diagonals(self):
        # comm_range < sqrt(2) excludes diagonals
        net = Network.grid(3, 3, spacing=1.0, comm_range=1.1)
        d0 = net.devices[0]
        assert len(d0.neighbors) == 2
        d4 = net.devices[4]
        assert len(d4.neighbors) == 4

    def test_remove_device(self):
        net = Network(comm_range=2.0)
        net.add_device((0, 0), device_id=0)
        net.add_device((1, 0), device_id=1)
        net.remove_device(0)
        assert 0 not in net.devices
        assert 1 in net.devices

    def test_random_factory(self):
        net = Network.random(20, width=5, height=5, comm_range=2.0,
                             rng=np.random.default_rng(0))
        assert len(net.devices) == 20
        # At least some devices should have neighbors
        has_neighbors = sum(1 for d in net.devices.values() if d.neighbors)
        assert has_neighbors > 0

    def test_distance(self):
        net = Network(comm_range=5.0)
        net.add_device((0, 0), device_id=0)
        net.add_device((3, 4), device_id=1)
        assert abs(net.get_distance(0, 1) - 5.0) < 1e-9
