"""Procedural 3D geometry engine - text prompts -> STL via trimesh."""

import os
import re
import numpy as np
import trimesh
from trimesh.path import encoding as path_encoding
from typing import List, Tuple, Union


def _extrude_path(path_2d, height):
    """Extrude a 2D path into a 3D mesh (works around trimesh 4.x API changes)."""
    try:
        return trimesh.creation.extrude(path_2d, height)
    except AttributeError:
        # Fallback for trimesh 4.x: use the path object directly
        return path_2d.extrude(height)


class GeometryEngine:
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_dim(self, text, default=1.0):
        """Extracts float from strings like '4x2' or 'r=3.0'."""
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return [float(n) for n in nums] if nums else [default]

    def create_primitive(self, prompt: str):
        """Parses a single shape prompt and returns a trimesh object."""
        p = prompt.lower().strip()

        if "box" in p or "cube" in p:
            dims = self._parse_dim(p)
            extents = dims if len(dims) >= 3 else (dims[0], dims[0], dims[0]) if len(dims)==1 else (dims[0], dims[1], 1.0)
            return trimesh.creation.box(extents=extents)

        if "sphere" in p:
            dims = self._parse_dim(p)
            r = dims[0] if dims else 1.0
            return trimesh.creation.uv_sphere(radius=r)

        if "cylinder" in p:
            dims = self._parse_dim(p)
            r = dims[0] if len(dims) >= 1 else 1.0
            h = dims[1] if len(dims) >= 2 else 2.0
            return trimesh.creation.cylinder(radius=r, height=h)

        if "cone" in p:
            dims = self._parse_dim(p)
            r = dims[0] if len(dims) >= 1 else 1.0
            h = dims[1] if len(dims) >= 2 else 2.0
            return trimesh.creation.cone(radius=r, height=h)

        if "torus" in p:
            dims = self._parse_dim(p)
            major = dims[0] if len(dims) >= 1 else 2.0
            minor = dims[1] if len(dims) >= 2 else 0.5
            return trimesh.creation.torus(major_radius=major, minor_radius=minor)

        if "tetra" in p:
            dims = self._parse_dim(p)
            s = dims[0] if dims else 1.0
            return trimesh.creation.regular_polyhedron(vertex_count=4, scale=s)

        if "octa" in p:
            dims = self._parse_dim(p)
            s = dims[0] if dims else 1.0
            return trimesh.creation.regular_polyhedron(vertex_count=6, scale=s)

        if "pentagon_prism" in p:
            dims = self._parse_dim(p)
            r = dims[0] if len(dims) >= 1 else 1.0
            h = dims[1] if len(dims) >= 2 else 2.0
            theta = np.linspace(0, 2*np.pi, 6)[:-1]
            vertices_2d = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            path = trimesh.path.entities.Polygon(vertices_2d)
            return _extrude_path(path, h)

        if "hex_prism" in p:
            dims = self._parse_dim(p)
            r = dims[0] if len(dims) >= 1 else 1.0
            h = dims[1] if len(dims) >= 2 else 2.0
            theta = np.linspace(0, 2*np.pi, 7)[:-1]
            vertices_2d = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            path = trimesh.path.entities.Polygon(vertices_2d)
            return _extrude_path(path, h)

        if "prism" in p:
            # Generic prism: rectangular box with aspect ratio
            dims = self._parse_dim(p)
            w = dims[0] if len(dims) >= 2 else dims[0] if dims else 2.0
            d = dims[1] if len(dims) >= 2 else 2.0
            h = dims[2] if len(dims) >= 3 else 2.0
            return trimesh.creation.box(extents=[w, d, h])

        return trimesh.creation.box(extents=[1, 1, 1])

    def generate(self, prompt: str):
        p = prompt.lower().strip()
        try:
            if "combine" in p:
                parts = p.replace("combine", "").split()
                op_idx = -1
                for op in ["union", "intersect", "difference"]:
                    if op in parts:
                        op_idx = parts.index(op)
                        break
                if op_idx != -1:
                    meshA_prompt = " ".join(parts[:op_idx])
                    op = parts[op_idx]
                    meshB_prompt = " ".join(parts[op_idx+1:])
                    a, b = self.create_primitive(meshA_prompt), self.create_primitive(meshB_prompt)
                    if op == "union": result = a.union(b)
                    elif op == "intersect": result = a.intersection(b)
                    else: result = a.difference(b)
                    return f"Combined {meshA_prompt} and {meshB_prompt} via {op}", result, "multi_boolean"

            if "stack on" in p:
                rem = p.split("stack")[1].split("on")
                a, b = self.create_primitive(rem[0].strip()), self.create_primitive(rem[1].strip())
                z_offset = (b.bounds[1][2] - b.bounds[0][2]) / 2 + (a.bounds[1][2] - a.bounds[0][2]) / 2
                a.apply_translation([0, 0, z_offset])
                return f"Stacked {rem[0].strip()} on {rem[1].strip()}", a.union(b), "multi_assembly"

            if "align beside" in p:
                rem = p.split("align")[1].split("beside")
                a, b = self.create_primitive(rem[0].strip()), self.create_primitive(rem[1].strip())
                x_offset = (a.bounds[1][0] - a.bounds[0][0]) / 2 + (b.bounds[1][0] - b.bounds[0][0]) / 2
                b.apply_translation([x_offset, 0, 0])
                return f"Aligned {rem[0].strip()} beside {rem[1].strip()}", a.union(b), "multi_assembly"

            mesh = self.create_primitive(p)
            return f"Generated {p}", mesh, "single"
        except Exception as e:
            return f"Error: {e}. Returning box.", trimesh.creation.box(), "error"

    def save(self, meshes, name):
        meshes_list = meshes if isinstance(meshes, list) else [meshes]
        combined = trimesh.util.concatenate(meshes_list)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
        out_path = os.path.join(self.output_dir, f"{safe_name}.stl")
        combined.export(out_path)
        return safe_name, [out_path], f"Vertices: {len(combined.vertices)}, Faces: {len(combined.faces)}"

    def export_step(self, meshes, name):
        """Export to STEP format using cadquery if available, otherwise skip."""
        try:
            import cadquery as cq
        except ImportError:
            return None, "STEP export requires 'cadquery' package"

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
        out_path = os.path.join(self.output_dir, f"{safe_name}.step")
        meshes_list = meshes if isinstance(meshes, list) else [meshes]
        combined = trimesh.util.concatenate(meshes_list)

        try:
            # Convert trimesh to cadquery Solid
            solid = cq.Solid.makeFromTessellation(combined)
            # Export to STEP
            cq.exporters.export(solid, out_path, fmt="STEP")
            return out_path, None
        except Exception as e:
            return None, f"cadquery export failed: {e}"
