/** three.js GLB viewer with orbit controls. */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export function GLBView({ url, title }: { url: string; title?: string }) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    let disposed = false;
    let scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer,
      controls: OrbitControls, raf = 0;

    scene = new THREE.Scene();
    scene.background = new THREE.Color("#0a0d11");
    camera = new THREE.PerspectiveCamera(50, holder.clientWidth / holder.clientHeight, 0.01, 2000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(holder.clientWidth, holder.clientHeight);
    holder.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(2, 4, 3);
    scene.add(dir);
    const grid = new THREE.GridHelper(10, 20, 0x2f3b4a, 0x1c232d);
    scene.add(grid);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        if (disposed) return;
        const root = gltf.scene;
        const box = new THREE.Box3().setFromObject(root);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        root.position.sub(center);
        scene.add(root);
        camera.position.set(center.x, center.y + maxDim * 0.25, center.z + maxDim * 1.6);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      (err) => { if (!disposed) setFailed(String((err as Error)?.message || err)); },
    );

    const loop = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    const ro = new ResizeObserver(() => {
      if (disposed || !holder.clientWidth) return;
      camera.aspect = holder.clientWidth / holder.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(holder.clientWidth, holder.clientHeight);
    });
    ro.observe(holder);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      holder.removeChild(renderer.domElement);
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        if (m.material) {
          const mats = Array.isArray(m.material) ? m.material : [m.material];
          mats.forEach((x) => (x as THREE.Material & { map?: THREE.Texture }).map?.dispose?.());
          mats.forEach((x) => (x as THREE.Material).dispose());
        }
      });
    };
  }, [url]);

  return (
    <div>
      {title && <div className="viz-caption">{title} · drag to orbit · scroll to zoom · right-drag to pan</div>}
      <div className="glbview" ref={holderRef} />
      {failed && <div className="note" style={{ color: "var(--err)" }}>GLB load failed: {failed}</div>}
    </div>
  );
}
