import {
  AmbientLight,
  DirectionalLight,
  PerspectiveCamera,
  Scene,
  Timer,
  WebGLRenderer,
} from "three";

import { PlaceholderCube } from "./scene/placeholderCube";

/**
 * Phase 0 viewer entry point: renders a spinning placeholder cube.
 *
 * Replay loading, the track mesh, cars and UI arrive in Phase 10.
 */
function start(container: HTMLElement): () => void {
  const renderer = new WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new Scene();
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);

  const ambient = new AmbientLight(0xffffff, 0.4);
  const sun = new DirectionalLight(0xffffff, 1.2);
  sun.position.set(3, 4, 5);
  scene.add(ambient, sun);

  const cube = new PlaceholderCube();
  scene.add(cube.object);

  const resize = (): void => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener("resize", resize);

  const timer = new Timer();
  renderer.setAnimationLoop((timeMs: number) => {
    timer.update(timeMs);
    cube.update(timer.getDelta());
    renderer.render(scene, camera);
  });

  return () => {
    renderer.setAnimationLoop(null);
    window.removeEventListener("resize", resize);
    timer.dispose();
    cube.dispose();
    sun.dispose();
    ambient.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  };
}

const container = document.getElementById("app");
if (container === null) {
  throw new Error("viewer: #app element not found");
}
const stop = start(container);
import.meta.hot?.dispose(stop);
