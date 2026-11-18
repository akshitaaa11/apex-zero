"""glTF export of a track (P2.7): road ribbon, kerbs, run-off plane, polylines.

Geometry (world frame, z up, metres):

- **Road**: a closed ribbon between the left and right boundaries at every 1 m ``s`` sample.
  The surface follows the bank angle: ``z(n) = z(s) + n tan(bank)``. Triangles are wound
  counter-clockwise seen from above, so face normals point up (glTF front faces are CCW).
  Vertex normals are area-weighted face normals. UVs tile an asphalt texture every
  ``tile_m`` metres (``u`` across the road, ``v`` along ``s``).
- **Kerbs**: strips ``kerb_width_m`` wide outside both boundaries wherever ``|kappa|`` exceeds
  ``kerb_kappa_min`` (corners), raised ``kerb_height_m``. Purely visual.
- **Run-off**: one flat ground plane under the whole track (bounding box plus a margin),
  ``0.3 m`` below the lowest road point.
- **Polylines**: centerline and both boundaries as ``LINE_LOOP`` primitives.

glTF is y-up, so world ``(x, y, z)`` maps to glTF ``(x, z, -y)`` (a rotation: winding and
handedness are preserved). Colours are plain materials; no logos or liveries.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygltflib as gl
from matplotlib import image as mpimg
from numpy.typing import NDArray

from apex_zero.track.asset import TrackAsset
from apex_zero.utils.seeding import make_rng

__all__ = ["MeshPart", "build_parts", "export_glb", "to_gltf_frame"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeshPart:
    """One primitive: world-frame vertices, optional normals/UVs, and indices."""

    name: str
    positions: FloatArray  # (V, 3) world frame
    indices: NDArray[np.uint32]  # (F, 3) triangles, or (V,) for a line loop
    normals: FloatArray | None = None
    uvs: FloatArray | None = None
    mode: int = gl.TRIANGLES
    colour: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    textured: bool = False


def to_gltf_frame(v: FloatArray) -> FloatArray:
    """World ``(x, y, z)`` (z up) -> glTF ``(x, z, -y)`` (y up)."""
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def _vertex_normals(pos: FloatArray, tris: NDArray[np.uint32]) -> FloatArray:
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)  # length = 2 * area
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    return vn


def _ribbon(inner: FloatArray, outer: FloatArray, closed: bool) -> NDArray[np.uint32]:
    """Triangles between two polylines of equal length; ``inner`` is the right-hand edge.

    Vertex layout: ``[outer_0, inner_0, outer_1, inner_1, ...]`` (outer = left edge).
    """
    n = len(inner)
    k = np.arange(n if closed else n - 1)
    k1 = (k + 1) % n
    left, right = 2 * k, 2 * k + 1
    left1, right1 = 2 * k1, 2 * k1 + 1
    t1 = np.column_stack([right, right1, left])
    t2 = np.column_stack([left, right1, left1])
    return np.vstack([t1, t2]).astype(np.uint32)


def _interleave(left: FloatArray, right: FloatArray) -> FloatArray:
    out = np.empty((2 * len(left), 3))
    out[0::2], out[1::2] = left, right
    return out


def _runs(mask: NDArray[np.bool_]) -> list[NDArray[np.int64]]:
    """Index runs where ``mask`` is True, joining a run that wraps over the loop seam."""
    if mask.all():
        return [np.arange(len(mask))]
    if not mask.any():
        return []
    start = int(np.argmin(mask))  # begin at a False so no run is split by the seam
    order = np.roll(np.arange(len(mask)), -start)
    m = mask[order]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(np.int8), [0]])))
    return [order[a:b] for a, b in zip(edges[0::2], edges[1::2], strict=True)]


def build_parts(
    asset: TrackAsset,
    *,
    tile_m: float = 4.0,
    kerb_kappa_min: float = 1 / 150,
    kerb_width_m: float = 1.0,
    kerb_height_m: float = 0.02,
    min_kerb_len_m: float = 10.0,
    runoff_margin_m: float = 100.0,
) -> list[MeshPart]:
    """World-frame mesh parts of a track (see module docstring)."""
    c = asset.xyz
    nrm = asset.line.normals
    tb = np.tan(asset.bank)

    def edge(n: FloatArray, dz: float = 0.0, idx: NDArray[np.int64] | None = None) -> FloatArray:
        sel = slice(None) if idx is None else idx
        p = c[sel].copy()
        p[:, :2] += nrm[sel] * n[:, None]
        p[:, 2] += n * tb[sel] + dz
        return p

    left, right = edge(asset.w_left), edge(-asset.w_right)
    pos = _interleave(left, right)
    tris = _ribbon(right, left, closed=True)
    width = asset.w_left + asset.w_right
    uv = np.empty((len(pos), 2))
    uv[0::2, 0], uv[1::2, 0] = width / tile_m, 0.0
    uv[0::2, 1] = uv[1::2, 1] = asset.s / tile_m
    parts = [
        MeshPart(
            "road",
            pos,
            tris,
            _vertex_normals(pos, tris),
            uv,
            colour=(0.35, 0.35, 0.37, 1.0),
            textured=True,
        )
    ]

    corner = np.abs(asset.kappa) > kerb_kappa_min
    kerb_pos, kerb_tris = [], []
    offset = 0
    for run in _runs(corner):
        if len(run) * asset.ds < min_kerb_len_m:
            continue
        for side in (+1, -1):
            w = asset.w_left if side > 0 else asset.w_right
            inner = edge(side * w[run], kerb_height_m, run)
            outer = edge(side * (w[run] + kerb_width_m), kerb_height_m, run)
            lft, rgt = (outer, inner) if side > 0 else (inner, outer)
            kerb_pos.append(_interleave(lft, rgt))
            kerb_tris.append(_ribbon(rgt, lft, closed=False) + offset)
            offset += 2 * len(run)
    if kerb_pos:
        kp, kt = np.vstack(kerb_pos), np.vstack(kerb_tris).astype(np.uint32)
        parts.append(
            MeshPart("kerbs", kp, kt, _vertex_normals(kp, kt), colour=(0.75, 0.12, 0.12, 1.0))
        )

    lo = np.minimum(left.min(axis=0), right.min(axis=0)) - runoff_margin_m
    hi = np.maximum(left.max(axis=0), right.max(axis=0)) + runoff_margin_m
    zg = float(min(left[:, 2].min(), right[:, 2].min()) - 0.3)
    gp = np.array([[lo[0], lo[1], zg], [hi[0], lo[1], zg], [hi[0], hi[1], zg], [lo[0], hi[1], zg]])
    gt = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)  # CCW from above
    parts.append(
        MeshPart("runoff", gp, gt, np.tile([0.0, 0.0, 1.0], (4, 1)), colour=(0.22, 0.36, 0.2, 1.0))
    )

    loop = np.arange(len(c), dtype=np.uint32)
    lift = np.array([0.0, 0.0, 0.05])
    parts.append(
        MeshPart("centerline", c + lift, loop, mode=gl.LINE_LOOP, colour=(1.0, 0.85, 0.1, 1.0))
    )
    parts.append(
        MeshPart("boundary_left", left + lift, loop, mode=gl.LINE_LOOP, colour=(1.0, 1.0, 1.0, 1.0))
    )
    parts.append(
        MeshPart(
            "boundary_right", right + lift, loop, mode=gl.LINE_LOOP, colour=(1.0, 1.0, 1.0, 1.0)
        )
    )
    return parts


def _asphalt_png(seed: int, size: int = 128) -> bytes:
    rng = make_rng(seed)
    noise = 0.30 + 0.08 * rng.standard_normal((size, size))
    img = np.clip(np.repeat(noise[..., None], 3, axis=2), 0.0, 1.0)
    buf = io.BytesIO()
    mpimg.imsave(buf, img, format="png")
    return buf.getvalue()


def export_glb(asset: TrackAsset, path: Path, *, texture_seed: int = 0) -> Path:
    """Write ``track.glb`` (binary glTF 2.0) for ``asset``."""
    parts = build_parts(asset)
    blob = bytearray()
    views: list[gl.BufferView] = []
    accessors: list[gl.Accessor] = []

    def add(
        data: NDArray[np.generic], ctype: int, atype: str, target: int | None, minmax: bool = False
    ) -> int:
        while len(blob) % 4:
            blob.append(0)
        raw = np.ascontiguousarray(data).tobytes()
        views.append(
            gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(raw), target=target)
        )
        blob.extend(raw)
        acc = gl.Accessor(
            bufferView=len(views) - 1, componentType=ctype, count=len(data), type=atype
        )
        if minmax:
            acc.min = np.asarray(data).min(axis=0).astype(float).tolist()
            acc.max = np.asarray(data).max(axis=0).astype(float).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    png = _asphalt_png(texture_seed)
    while len(blob) % 4:
        blob.append(0)
    img_view = len(views)
    views.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(png)))
    blob.extend(png)

    materials, meshes, nodes = [], [], []
    for part in parts:
        mat = gl.Material(
            name=part.name,
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorFactor=list(part.colour),
                metallicFactor=0.0,
                roughnessFactor=0.9,
                baseColorTexture=gl.TextureInfo(index=0) if part.textured else None,
            ),
            doubleSided=False,
        )
        materials.append(mat)
        pos = to_gltf_frame(part.positions).astype(np.float32)
        attrs = gl.Attributes(POSITION=add(pos, gl.FLOAT, gl.VEC3, gl.ARRAY_BUFFER, True))
        if part.normals is not None:
            attrs.NORMAL = add(
                to_gltf_frame(part.normals).astype(np.float32), gl.FLOAT, gl.VEC3, gl.ARRAY_BUFFER
            )
        if part.uvs is not None:
            attrs.TEXCOORD_0 = add(part.uvs.astype(np.float32), gl.FLOAT, gl.VEC2, gl.ARRAY_BUFFER)
        idx = add(
            part.indices.reshape(-1).astype(np.uint32),
            gl.UNSIGNED_INT,
            gl.SCALAR,
            gl.ELEMENT_ARRAY_BUFFER,
        )
        meshes.append(
            gl.Mesh(
                name=part.name,
                primitives=[
                    gl.Primitive(
                        attributes=attrs, indices=idx, material=len(materials) - 1, mode=part.mode
                    )
                ],
            )
        )
        nodes.append(gl.Node(name=part.name, mesh=len(meshes) - 1))

    gltf = gl.GLTF2(
        asset=gl.Asset(version="2.0", generator="apex-zero track/mesh.py"),
        scene=0,
        scenes=[gl.Scene(name=asset.name, nodes=list(range(len(nodes))))],
        nodes=nodes,
        meshes=meshes,
        materials=materials,
        accessors=accessors,
        bufferViews=views,
        buffers=[gl.Buffer(byteLength=len(blob))],
        images=[gl.Image(bufferView=img_view, mimeType="image/png", name="asphalt")],
        samplers=[
            gl.Sampler(
                magFilter=gl.LINEAR,
                minFilter=gl.LINEAR_MIPMAP_LINEAR,
                wrapS=gl.REPEAT,
                wrapT=gl.REPEAT,
            )
        ],
        textures=[gl.Texture(sampler=0, source=0)],
        extras={"track": asset.name, "build_hash": asset.build_hash, "length_m": asset.length},
    )
    gltf.set_binary_blob(bytes(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(path))
    return path
