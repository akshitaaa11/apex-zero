from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib as gl
import pytest
import trimesh

from apex_zero.track.mesh import _runs, build_parts, export_glb, to_gltf_frame
from tests.unit.track.test_components import make_asset


@pytest.fixture
def asset():  # type: ignore[no-untyped-def]
    n = 100
    kappa = np.where((np.arange(n) > 10) & (np.arange(n) < 30), 0.02, 0.001)
    return make_asset(kappa=kappa, bank=np.full(n, np.deg2rad(3.0)), z=np.linspace(0, 5, n))


def test_gltf_frame_is_a_rotation() -> None:
    m = to_gltf_frame(np.eye(3))
    assert np.linalg.det(m) == pytest.approx(1.0)
    np.testing.assert_allclose(to_gltf_frame(np.array([[1.0, 2.0, 3.0]])), [[1.0, 3.0, -2.0]])


def test_road_normals_up_and_ribbon_closed(asset) -> None:  # type: ignore[no-untyped-def]
    parts = {p.name: p for p in build_parts(asset)}
    road = parts["road"]
    tri = trimesh.Trimesh(road.positions, road.indices, process=False)
    assert (tri.face_normals[:, 2] > 0).all()
    assert (road.normals[:, 2] > 0).all()
    open_edges = trimesh.grouping.group_rows(tri.edges_sorted, require_count=1)
    assert len(open_edges) == 2 * len(asset.s)  # two boundary loops, no seam
    # banked: the left edge (n > 0) is higher by w * tan(bank)
    dz = road.positions[0::2, 2] - road.positions[1::2, 2]
    np.testing.assert_allclose(dz, 12.0 * np.tan(np.deg2rad(3.0)), atol=1e-9)
    assert road.uvs is not None and road.uvs[:, 1].max() == pytest.approx(asset.s[-1] / 4.0)


def test_kerbs_only_in_corners_and_facing_up(asset) -> None:  # type: ignore[no-untyped-def]
    parts = {p.name: p for p in build_parts(asset)}
    kerbs = parts["kerbs"]
    tri = trimesh.Trimesh(kerbs.positions, kerbs.indices, process=False)
    assert (tri.face_normals[:, 2] > 0).all()
    assert len(kerbs.positions) == 2 * 2 * 19  # both sides of the 19-sample corner
    assert {"runoff", "centerline", "boundary_left", "boundary_right"} <= set(parts)
    assert parts["runoff"].positions[:, 2].max() < asset.z.min()


def test_runs_join_across_seam() -> None:
    m = np.array([True, True, False, False, True, False, True])
    runs = [r.tolist() for r in _runs(m)]
    assert sorted(runs) == [[4], [6, 0, 1]]
    assert _runs(np.zeros(4, bool)) == [] and len(_runs(np.ones(4, bool))[0]) == 4


def test_export_glb_loads_and_is_valid_gltf(asset, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = export_glb(asset, tmp_path / "track.glb")
    g = gl.GLTF2().load_binary(str(path))
    assert g.asset.version == "2.0"
    assert [n.name for n in g.nodes] == [
        "road",
        "kerbs",
        "runoff",
        "centerline",
        "boundary_left",
        "boundary_right",
    ]
    modes = {m.name: m.primitives[0].mode for m in g.meshes}
    assert modes["centerline"] == gl.LINE_LOOP and modes["road"] == gl.TRIANGLES
    pos_acc = g.accessors[g.meshes[0].primitives[0].attributes.POSITION]
    assert pos_acc.min is not None and pos_acc.max is not None
    scene = trimesh.load(path)
    road = scene.geometry["road"]
    assert (road.face_normals[:, 1] > 0).all()  # y is up in glTF
    assert g.extras["build_hash"] == asset.build_hash
    assert export_glb(asset, tmp_path / "again.glb").read_bytes() == path.read_bytes()
