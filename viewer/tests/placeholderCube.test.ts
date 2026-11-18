import { describe, expect, it, vi } from "vitest";

import { advanceAngle, CUBE_SPIN_RAD_PER_S, PlaceholderCube } from "../src/scene/placeholderCube";

describe("advanceAngle", () => {
  it("advances by rate * dt", () => {
    expect(advanceAngle(0, 2, 0.25)).toBeCloseTo(0.5);
  });

  it("wraps into [0, 2π)", () => {
    const a = advanceAngle(2 * Math.PI - 0.1, 1, 0.2);
    expect(a).toBeCloseTo(0.1);
    const b = advanceAngle(0.1, -1, 0.2);
    expect(b).toBeCloseTo(2 * Math.PI - 0.1);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThan(2 * Math.PI);
  });
});

describe("PlaceholderCube", () => {
  it("spins at the configured rate", () => {
    const cube = new PlaceholderCube();
    cube.update(0.5);
    expect(cube.mesh.rotation.x).toBeCloseTo(CUBE_SPIN_RAD_PER_S.x * 0.5);
    expect(cube.mesh.rotation.y).toBeCloseTo(CUBE_SPIN_RAD_PER_S.y * 0.5);
    cube.dispose();
  });

  it("disposes its geometry and material", () => {
    const cube = new PlaceholderCube();
    const geometryDispose = vi.spyOn(cube.mesh.geometry, "dispose");
    const materialDispose = vi.spyOn(cube.mesh.material, "dispose");
    cube.dispose();
    expect(geometryDispose).toHaveBeenCalledOnce();
    expect(materialDispose).toHaveBeenCalledOnce();
  });
});
