import { BoxGeometry, Mesh, MeshStandardMaterial, type Object3D } from "three";

/** Rotation speed of the placeholder cube, radians per second about (x, y). */
export const CUBE_SPIN_RAD_PER_S = { x: 0.5, y: 0.8 } as const;

/**
 * Advance an angle by `rateRadPerS * dtS`, wrapped to [0, 2π).
 *
 * Pure so the render loop stays allocation-free and the maths is unit-testable.
 */
export function advanceAngle(angleRad: number, rateRadPerS: number, dtS: number): number {
  const twoPi = 2 * Math.PI;
  const next = (angleRad + rateRadPerS * dtS) % twoPi;
  return next < 0 ? next + twoPi : next;
}

/**
 * Phase 0 placeholder scene content: a spinning cube.
 *
 * Owns its geometry and material and disposes both in {@link dispose} (AGENTS.md §5).
 */
export class PlaceholderCube {
  readonly mesh: Mesh<BoxGeometry, MeshStandardMaterial>;

  constructor(sizeM = 1) {
    const geometry = new BoxGeometry(sizeM, sizeM, sizeM);
    const material = new MeshStandardMaterial({ color: 0xe10600, roughness: 0.4 });
    this.mesh = new Mesh(geometry, material);
  }

  get object(): Object3D {
    return this.mesh;
  }

  /** Advance the spin by `dtS` seconds. Allocates nothing. */
  update(dtS: number): void {
    const r = this.mesh.rotation;
    r.x = advanceAngle(r.x, CUBE_SPIN_RAD_PER_S.x, dtS);
    r.y = advanceAngle(r.y, CUBE_SPIN_RAD_PER_S.y, dtS);
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}
