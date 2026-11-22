"""
integration/generation/model3d.py — Core 3D Geometry Models, Hierarchy & Parsers.

Provides:
- Mesh3D: Vertices, faces, normals, vertex colors, material shading parameters.
- ModelNode3D: Component hierarchy graph with local position, rotation, scale.
- Model3D: Unified 3D representation with animation functions and component access.
- Native Zero-Dependency GLB (binary glTF 2.0) parser.
- Native Wavefront OBJ parser.
- 4x4 Transformation Matrix Math (Translation, Euler rotation, Scale, MVP).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Material3D:
    """Material shading parameters."""
    diffuse_color: Tuple[int, int, int] = (200, 200, 200)  # BGR
    ambient_coeff: float = 0.25
    diffuse_coeff: float = 0.75
    specular_coeff: float = 0.2
    shininess: float = 16.0
    roughness: float = 0.7
    metallic: float = 0.0
    opacity: float = 1.0
    wireframe: bool = False
    double_sided: bool = True


@dataclass
class Mesh3D:
    """Represents a 3D polygonal mesh."""
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray     # (M, 3) int32
    normals: Optional[np.ndarray] = None  # (M, 3) or (N, 3) float32
    colors: Optional[np.ndarray] = None   # (M, 3) or (N, 3) uint8 BGR
    material: Material3D = field(default_factory=Material3D)
    name: str = "mesh"

    def compute_face_normals(self) -> np.ndarray:
        """Computes normalized surface normals for all triangular faces."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        self.normals = (cross / norm).astype(np.float32)
        return self.normals

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center_and_normalize(self, target_radius: float = 1.0) -> None:
        """Centers mesh at origin and scales to target radius."""
        if len(self.vertices) == 0:
            return
        b_min, b_max = self.bounding_box()
        center = (b_min + b_max) * 0.5
        self.vertices -= center
        extent = np.linalg.norm(self.vertices, axis=1).max()
        if extent > 1e-6:
            self.vertices *= (target_radius / extent)


@dataclass
class ModelNode3D:
    """Node in a 3D component hierarchy."""
    name: str
    mesh: Optional[Mesh3D] = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # Euler [rx, ry, rz] radians
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    children: List[ModelNode3D] = field(default_factory=list)
    parent: Optional[ModelNode3D] = None
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: ModelNode3D) -> None:
        child.parent = self
        self.children.append(child)

    def get_local_transform(self) -> np.ndarray:
        return make_transform_matrix(self.position, self.rotation, self.scale)

    def get_world_transform(self) -> np.ndarray:
        local_mat = self.get_local_transform()
        if self.parent is not None:
            return self.parent.get_world_transform() @ local_mat
        return local_mat


@dataclass
class Model3D:
    """Complete structured 3D model containing nodes, animation, and metadata."""
    concept: str
    root: ModelNode3D
    visual_id: Optional[str] = None
    version: int = 1
    animation_fn: Optional[Callable[[Model3D, float], None]] = None
    category: str = "3D_OBJECT"  # "3D_OBJECT" | "SCIENTIFIC_MODEL" | "TECHNICAL_ASSEMBLY"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    animation_suspended: bool = False

    def get_all_meshes_with_transforms(self, t: float = 0.0) -> List[Tuple[Mesh3D, np.ndarray]]:
        """Applies animation tick and traverses hierarchy to collect meshes with world matrices."""
        if self.animation_fn is not None and not self.animation_suspended:
            try:
                self.animation_fn(self, t)
            except Exception:
                pass

        results: List[Tuple[Mesh3D, np.ndarray]] = []

        def _traverse(node: ModelNode3D, parent_mat: np.ndarray) -> None:
            if not node.visible:
                return
            world_mat = parent_mat @ node.get_local_transform()
            if node.mesh is not None:
                results.append((node.mesh, world_mat))
            for child in node.children:
                _traverse(child, world_mat)

        _traverse(self.root, np.eye(4, dtype=np.float32))
        return results

    def find_node(self, name: str) -> Optional[ModelNode3D]:
        """Finds a component node by name in the hierarchy."""
        stack = [self.root]
        while stack:
            cur = stack.pop()
            if cur.name == name:
                return cur
            stack.extend(cur.children)
        return None

    @property
    def num_triangles(self) -> int:
        """Returns the total number of triangular faces in this model."""
        return sum(len(m.faces) for m, _ in self.get_all_meshes_with_transforms())


# ---------------------------------------------------------------------------
# 4x4 Matrix Transformation Utilities
# ---------------------------------------------------------------------------

def make_translation_matrix(t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = t[0]
    m[1, 3] = t[1]
    m[2, 3] = t[2]
    return m


def make_rotation_matrix(r: np.ndarray) -> np.ndarray:
    rx, ry, rz = r[0], r[1], r[2]
    # Roll, pitch, yaw
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    mx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float32)
    my = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    return mz @ my @ mx


def make_scale_matrix(s: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def make_transform_matrix(pos: np.ndarray, rot: np.ndarray, scl: np.ndarray) -> np.ndarray:
    return make_translation_matrix(pos) @ make_rotation_matrix(rot) @ make_scale_matrix(scl)


def make_perspective_matrix(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def make_look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = eye - target
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    r = np.eye(4, dtype=np.float32)
    r[0, :3] = x_axis
    r[1, :3] = y_axis
    r[2, :3] = z_axis

    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye

    return r @ t


# ---------------------------------------------------------------------------
# Native GLB (Binary glTF 2.0) Parser (Zero-dependency)
# ---------------------------------------------------------------------------

def load_glb(file_path: Union[str, Path]) -> Model3D:
    """Load a GLB 2.0 scene into the existing Model3D hierarchy.

    Supports embedded buffer data, triangle primitives, node TRS/matrices,
    base-color factors, normals, and STEP/LINEAR node translation/rotation/scale
    animation channels. External buffers and skinning are intentionally unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"GLB file not found: {path}")
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("Invalid GLB: header or JSON chunk is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError("Invalid GLB 2.0 header or declared length")
    offset, doc, binary = 12, None, b""
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, offset); offset += 8
        if offset + size > len(data):
            raise ValueError("GLB chunk exceeds file length")
        chunk = data[offset:offset + size]; offset += size
        if kind == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8").rstrip(" \0"))
        elif kind == 0x004E4942:
            binary = chunk
    if doc is None or doc.get("asset", {}).get("version", "2.0").split(".")[0] != "2":
        raise ValueError("GLB has no valid glTF 2.0 JSON chunk")
    buffers = doc.get("buffers", [])
    if any(buf.get("uri") for buf in buffers):
        raise ValueError("External glTF buffers are not supported by load_glb")
    if buffers and buffers[0].get("byteLength", 0) > len(binary):
        raise ValueError("GLB binary chunk is shorter than buffer byteLength")
    component = {5120:('i1',1),5121:('u1',1),5122:('<i2',2),5123:('<u2',2),5125:('<u4',4),5126:('<f4',4)}
    width = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}
    def accessor(index):
        acc=doc["accessors"][index]
        if "bufferView" not in acc or acc.get("sparse"):
            raise ValueError("Sparse or bufferless accessors are unsupported")
        view=doc["bufferViews"][acc["bufferView"]]
        if view.get("buffer",0) != 0: raise ValueError("Only the embedded GLB buffer is supported")
        dtype,size=component[acc["componentType"]]; n=width[acc["type"]]
        stride=view.get("byteStride",size*n); start=view.get("byteOffset",0)+acc.get("byteOffset",0)
        if stride == size*n:
            arr=np.frombuffer(binary,dtype=dtype,count=acc["count"]*n,offset=start).reshape(acc["count"],n)
        else:
            arr=np.ndarray((acc["count"],n),dtype=dtype,buffer=binary,offset=start,strides=(stride,size)).copy()
        if acc.get("normalized") and acc["componentType"] != 5126:
            if acc["componentType"] in (5120,5122):
                info=np.iinfo(arr.dtype); arr=np.maximum(arr.astype(np.float32)/info.max,-1.0)
            else: arr=arr.astype(np.float32)/np.iinfo(arr.dtype).max
        return arr[:,0] if n == 1 else arr
    materials=[]
    for m in doc.get("materials",[]):
        pbr=m.get("pbrMetallicRoughness",{}); rgba=pbr.get("baseColorFactor",[1,1,1,1])
        materials.append(Material3D(diffuse_color=tuple(int(round(float(c)*255)) for c in (rgba[2],rgba[1],rgba[0])), opacity=float(rgba[3]), double_sided=bool(m.get("doubleSided",False))))
    meshes=[]
    for mi,m in enumerate(doc.get("meshes",[])):
        prims=[]
        for pi,prim in enumerate(m.get("primitives",[])):
            if prim.get("mode",4) != 4: continue
            attrs=prim.get("attributes",{})
            if "POSITION" not in attrs: continue
            verts=np.asarray(accessor(attrs["POSITION"]),dtype=np.float32)
            normals=np.asarray(accessor(attrs["NORMAL"]),dtype=np.float32) if "NORMAL" in attrs else None
            indices=np.asarray(accessor(prim["indices"]),dtype=np.int64).reshape(-1) if "indices" in prim else np.arange(len(verts))
            if len(indices)%3: raise ValueError("Triangle index count must be divisible by three")
            faces=indices.reshape(-1,3).astype(np.int32)
            if faces.size and (faces.min()<0 or faces.max()>=len(verts)): raise ValueError("Primitive index out of range")
            mat=materials[prim["material"]] if "material" in prim else Material3D()
            mesh=Mesh3D(verts,faces,normals=normals,material=mat,name=f"{m.get('name',f'mesh_{mi}')}_p{pi}")
            if mesh.normals is None: mesh.compute_face_normals()
            prims.append(mesh)
        meshes.append(prims)
    nodes=[]
    for i,n in enumerate(doc.get("nodes",[])):
        if "matrix" in n:
            matrix=np.asarray(n["matrix"],dtype=np.float32).reshape(4,4,order="F")
            # Decompose glTF TRS into representation fields; reject shear.
            scale=np.linalg.norm(matrix[:3,:3],axis=0); rot=matrix[:3,:3]/np.where(scale==0,1,scale)
            if not np.allclose(rot.T@rot,np.eye(3),atol=1e-4): raise ValueError("Sheared node matrices are unsupported")
            sy=-rot[2,0]
            if abs(sy) < 0.999999:
                euler=np.array([math.atan2(rot[2,1],rot[2,2]),math.asin(sy),math.atan2(rot[1,0],rot[0,0])],dtype=np.float32)
            else:
                euler=np.array([math.atan2(-rot[1,2],rot[1,1]),math.copysign(math.pi/2,sy),0.0],dtype=np.float32)
            pos=matrix[:3,3]
        else:
            pos=np.asarray(n.get("translation",[0,0,0]),dtype=np.float32)
            scale=np.asarray(n.get("scale",[1,1,1]),dtype=np.float32)
            q=np.asarray(n.get("rotation",[0,0,0,1]),dtype=np.float64)
            x,y,z,w=q
            rot=np.array([
                [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],
            ],dtype=np.float64)
            sy=-rot[2,0]
            if abs(sy) < 0.999999:
                euler=np.array([math.atan2(rot[2,1],rot[2,2]),math.asin(sy),math.atan2(rot[1,0],rot[0,0])],dtype=np.float32)
            else:
                euler=np.array([math.atan2(-rot[1,2],rot[1,1]),math.copysign(math.pi/2,sy),0.0],dtype=np.float32)
        node=ModelNode3D(n.get("name",f"node_{i}"),position=pos,rotation=euler,scale=np.asarray(scale,dtype=np.float32),metadata={"gltf_node":i})
        if "mesh" in n:
            for mesh in meshes[n["mesh"]]:
                child=ModelNode3D(mesh.name,mesh=mesh); node.add_child(child)
        nodes.append(node)
    for i,n in enumerate(doc.get("nodes",[])):
        for child in n.get("children",[]): nodes[i].add_child(nodes[child])
    scene_idx=doc.get("scene",0); scene=doc.get("scenes",[])
    roots=[nodes[i] for i in (scene[scene_idx].get("nodes",[]) if scene else range(len(nodes))) if i < len(nodes)]
    root=ModelNode3D(path.stem)
    for node in roots: root.add_child(node)
    if not nodes:
        for primitives in meshes:
            for mesh in primitives: root.add_child(ModelNode3D(mesh.name, mesh=mesh))
    animations=[]
    for anim in doc.get("animations",[]):
        for channel in anim.get("channels",[]):
            target=channel.get("target",{}); ni=target.get("node"); path_name=target.get("path")
            if ni is None or ni>=len(nodes) or path_name not in ("translation","rotation","scale"): continue
            sampler=anim["samplers"][channel["sampler"]]; times=np.asarray(accessor(sampler["input"]),dtype=np.float32).reshape(-1)
            values=np.asarray(accessor(sampler["output"]),dtype=np.float32)
            animations.append((nodes[ni],path_name,times,values,sampler.get("interpolation","LINEAR")))
    original=[(n,n.position.copy(),n.rotation.copy(),n.scale.copy()) for n in nodes]
    def animate(model,t):
        for n,p,r,scl in original: n.position=p.copy(); n.rotation=r.copy(); n.scale=scl.copy()
        for node,prop,times,values,interp in animations:
            if not len(times): continue
            x=float(np.clip(t,times[0],times[-1])); j=int(np.searchsorted(times,x,side="right"))
            if j==0: value=values[0]
            elif j>=len(times) or interp=="STEP": value=values[min(j-1,len(values)-1)]
            else:
                u=(x-times[j-1])/(times[j]-times[j-1]); value=values[j-1]*(1-u)+values[j]*u
            if prop=="rotation":
                q=np.asarray(value); xq,yq,zq,wq=q
                value=np.array([math.atan2(2*(wq*xq+yq*zq),1-2*(xq*xq+yq*yq)),math.asin(np.clip(2*(wq*yq-zq*xq),-1,1)),math.atan2(2*(wq*zq+xq*yq),1-2*(yq*yq+zq*zq))])
            setattr(node,"position" if prop=="translation" else prop,np.asarray(value,dtype=np.float32))
    model=Model3D(concept=path.stem,root=root,category="3D_OBJECT",description=f"Loaded from GLB: {path.name}",animation_fn=animate if animations else None,metadata={"gltf_animations":len(doc.get("animations",[])),"unsupported_textures":len(doc.get("textures",[])),"unsupported_skins":len(doc.get("skins",[]))})
    return model


# ---------------------------------------------------------------------------
# Native Wavefront OBJ Parser
# ---------------------------------------------------------------------------

def load_obj(file_path: Union[str, Path]) -> Model3D:
    """Parses a Wavefront .obj file into a Model3D object."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {path}")

    verts: List[List[float]] = []
    faces: List[List[int]] = []
    normals: List[List[float]] = []

    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vn" and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Handle v, v/vt, v/vt/vn, v//vn
                face_indices = []
                for p in parts[1:4]:
                    idx_str = p.split("/")[0]
                    face_indices.append(int(idx_str) - 1)
                faces.append(face_indices)

    v_arr = np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)
    f_arr = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    mesh = Mesh3D(vertices=v_arr, faces=f_arr, name=path.stem)
    if len(f_arr) > 0:
        mesh.compute_face_normals()
    mesh.center_and_normalize(1.0)

    root = ModelNode3D(name=path.stem, mesh=mesh)
    return Model3D(concept=path.stem, root=root, category="3D_OBJECT")





