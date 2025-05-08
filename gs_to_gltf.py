import argparse
import base64
import json
import numpy as np
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from plyfile import PlyData
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
from pymap3d import enu2ecef
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from typing import List, Dict, Tuple


# NumpyEncoder Used to serialize NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Define the data structure of the point
class Point:
    __slots__ = ["position", "color", "scale", "rotation"]

    def __init__(
        self,
        position: Tuple[float, float, float],
        color: Tuple[int, int, int, int],
        scale: Tuple[float, float, float],
        rotation: Tuple[int, int, int, int],
    ):
        self.position = position
        self.color = color
        self.scale = scale
        self.rotation = rotation

    def to_bytes(self) -> bytes:
        return struct.pack("3f4B3f4B", *self.position, *self.color, *self.scale, *self.rotation)

    @classmethod
    def from_bytes(cls, data: bytes):
        unpacked = struct.unpack("3f4B3f4B", data)
        position = unpacked[:3]
        color = unpacked[3:7]
        scale = unpacked[7:10]
        rotation = unpacked[10:]
        return cls(position, color, scale, rotation)


def read_splat_file(file_path: str) -> List[Point]:
    points = []
    with open(file_path, "rb") as f:
        while True:
            position_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            if not position_data:
                break
            position = struct.unpack("3f", position_data)
            scale = struct.unpack("3f", f.read(3 * 4))
            color = struct.unpack("4B", f.read(4 * 1))
            rotation_bytes = struct.unpack("4B", f.read(4))

            # Adjust the order of quaternions to (x, y, z, w)
            rotation = (rotation_bytes[1], rotation_bytes[2], rotation_bytes[3], rotation_bytes[0])
            points.append(Point(position, color, scale, rotation))
    return points


def read_ply_file(file_path: str) -> List[Point]:
    ply_data = PlyData.read(file_path)
    vertices = ply_data["vertex"]

    # Vectorization processing
    pos = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    f_dc = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
    opacity = vertices["opacity"].astype(np.float32)
    scales = np.exp(np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T)
    rot = np.vstack([vertices["rot_1"], vertices["rot_2"], vertices["rot_3"], vertices["rot_0"]]).T

    # Color processing
    SH_C0 = 0.28209479177387814
    colors = (0.5 + f_dc * SH_C0) * 255
    colors = np.clip(np.hstack([colors, (1 / (1 + np.exp(-opacity)))[:, None] * 255]), 0, 255).astype(np.uint8)

    # Rotating processing
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rotations = (rot / rot_norm * 128 + 128).clip(0, 255).astype(np.uint8)

    # Bulk generation of point objects
    return [Point(tuple(p), tuple(c), tuple(s), tuple(r)) for p, c, s, r in zip(pos, colors, scales, rotations)]


class OctreeNode:
    __slots__ = ["min", "max", "capacity", "points", "children", "_epsilon"]

    def __init__(self, bounds: Tuple[float, float, float, float, float, float], capacity: int):
        self.min = np.array(bounds[:3], dtype=np.float32)
        self.max = np.array(bounds[3:], dtype=np.float32)
        self.capacity = capacity
        self.points = []
        self.children = None
        self._epsilon = 1e-8

    def insert_batch(self, positions: np.ndarray, points_data: np.ndarray):
        # Calculate valid point mask
        in_bounds_mask = (
            (positions[:, 0] >= self.min[0] - self._epsilon)
            & (positions[:, 0] <= self.max[0] + self._epsilon)
            & (positions[:, 1] >= self.min[1] - self._epsilon)
            & (positions[:, 1] <= self.max[1] + self._epsilon)
            & (positions[:, 2] >= self.min[2] - self._epsilon)
            & (positions[:, 2] <= self.max[2] + self._epsilon)
        )

        valid_points = points_data[in_bounds_mask]

        if len(valid_points) == 0:
            return

        if self.children is None:
            if len(self.points) + len(valid_points) <= self.capacity:
                self.points.extend(valid_points)
                return
            self._subdivide(positions[in_bounds_mask], valid_points)
        else:
            self._distribute_to_children(positions[in_bounds_mask], valid_points)

    def _subdivide(self, positions: np.ndarray, points_data: np.ndarray):
        mid = (self.min + self.max) / 2
        self.children = [
            OctreeNode((self.min[0], self.min[1], self.min[2], mid[0], mid[1], mid[2]), self.capacity),
            OctreeNode((mid[0], self.min[1], self.min[2], self.max[0], mid[1], mid[2]), self.capacity),
            OctreeNode((self.min[0], mid[1], self.min[2], mid[0], self.max[1], mid[2]), self.capacity),
            OctreeNode((mid[0], mid[1], self.min[2], self.max[0], self.max[1], mid[2]), self.capacity),
            OctreeNode((self.min[0], self.min[1], mid[2], mid[0], mid[1], self.max[2]), self.capacity),
            OctreeNode((mid[0], self.min[1], mid[2], self.max[0], mid[1], self.max[2]), self.capacity),
            OctreeNode((self.min[0], mid[1], mid[2], mid[0], self.max[1], self.max[2]), self.capacity),
            OctreeNode((mid[0], mid[1], mid[2], self.max[0], self.max[1], self.max[2]), self.capacity),
        ]

        # Calculate the index of the child nodes to which each point belongs
        centroids = positions >= mid  # (N,3) bool array

        indices = centroids[:, 0].astype(int) * 4 + centroids[:, 1].astype(int) * 2 + centroids[:, 2].astype(int)

        for i, child in enumerate(self.children):
            mask = indices == i
            # Before inserting data for each child node, check whether the child node capacity exceeds the limit
            if len(child.points) + len(points_data[mask]) <= child.capacity:
                child.points.extend(points_data[mask])
            else:
                # If the child node capacity exceeds the limit, subdivision
                child.insert_batch(positions[mask], points_data[mask])

        # Clear the current node data
        self.points.clear()

    def _distribute_to_children(self, positions: np.ndarray, points_data: np.ndarray):
        mid = (self.min + self.max) / 2
        centroids = positions >= mid
        indices = centroids[:, 0].astype(int) * 4 + centroids[:, 1].astype(int) * 2 + centroids[:, 2].astype(int)

        for i, child in enumerate(self.children):
            mask = indices == i
            child.insert_batch(positions[mask], points_data[mask])

    def get_all_points(self) -> List[Point]:
        points = []
        stack = [self]
        while stack:
            node = stack.pop()
            points.extend(node.points)
            if node.children:
                stack.extend(node.children)
        return points


# Convert data to glTF files
def splat_to_gltf_with_gaussian_extension(points: List[Point], output_path: str):
    positions = np.array([p.position for p in points], dtype=np.float32)
    colors = np.array([p.color for p in points], dtype=np.uint8)
    scales = np.array([p.scale for p in points], dtype=np.float32)
    rotations = np.array([p.rotation for p in points], dtype=np.uint8)
    normalized_rotations = ((rotations - 128.0) / 128.0).astype(np.float32)

    gltf = GLTF2()
    gltf.extensionsUsed = ["primitive"]

    buffer = Buffer()
    gltf.buffers.append(buffer)

    positions_binary = positions.tobytes()
    colors_binary = colors.tobytes()
    scales_binary = scales.tobytes()
    rotations_binary = normalized_rotations.tobytes()

    def create_buffer_view(byte_offset: int, data: bytes, target: int = 34962) -> BufferView:
        return BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data), target=target)

    def create_accessor(
        buffer_view: int, component_type: int, count: int, type: str, max: List[float] = None, min: List[float] = None, normalized: bool = False
    ) -> Accessor:
        return Accessor(bufferView=buffer_view, componentType=component_type, count=count, type=type, max=max, min=min, normalized=normalized)

    buffer_views = [
        create_buffer_view(0, positions_binary),
        create_buffer_view(len(positions_binary), colors_binary),
        create_buffer_view(len(positions_binary) + len(colors_binary), rotations_binary),
        create_buffer_view(len(positions_binary) + len(colors_binary) + len(rotations_binary), scales_binary),
    ]

    accessors = [
        create_accessor(0, 5126, len(positions), "VEC3", positions.max(axis=0).tolist(), positions.min(axis=0).tolist()),
        create_accessor(1, 5121, len(colors), "VEC4", normalized=True),
        create_accessor(2, 5126, len(rotations), "VEC4"),
        create_accessor(3, 5126, len(scales), "VEC3"),
    ]
    gltf.bufferViews.extend(buffer_views)
    gltf.accessors.extend(accessors)

    primitive = Primitive(
        attributes={"POSITION": 0, "COLOR_0": 1, "_ROTATION": 2, "_SCALE": 3},
        mode=0,
        extensions={"KHR_gaussian_splatting": {"positions": 0, "colors": 1, "rotations": 2, "scales": 3}},
    )

    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)

    node = Node(mesh=0)
    gltf.nodes.append(node)
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    for material in gltf.materials:
        material.doubleSided = True

    gltf.buffers[0].uri = "data:application/octet-stream;base64," + base64.b64encode(
        positions_binary + colors_binary + rotations_binary + scales_binary
    ).decode("utf-8")
    gltf.save(output_path)
    print(f"glTF saved to: {output_path}")


# Generate 3D Tiles
def generate_3dtiles(node: OctreeNode, output_dir: str, tile_name: str):
    if node.children is not None:
        for i, child in enumerate(node.children):
            generate_3dtiles(child, output_dir, f"{tile_name}_{i}")
    elif len(node.points) > 0:
        points = node.get_all_points()
        splat_to_gltf_with_gaussian_extension(points, f"{output_dir}/{tile_name}.gltf")


# Generate transform
def create_transform_matrix_wkt_origin(wkt: str, wsk_origin: str) -> List[float]:
    # parse srs origin
    origin = tuple(map(float, wsk_origin.split(",")))
    origin_x, origin_y, origin_z = origin

    # Define the source coordinate system wkt
    source_wkt = wkt

    # Create a coordinate system converter (add exact conversion parameters)
    transformer = Transformer.from_crs(
        crs_from=CRS.from_wkt(source_wkt),
        crs_to=CRS.from_epsg(4978),
        always_xy=True,
        # accuracy=0.001 # Improve conversion accuracy
    )

    # Calculate the actual origin (projection coordinate system origin + SRSOrigin offset)
    actual_origin_proj = (origin_x, origin_y, origin_z)

    # Convert the actual origin to ecef
    X0, Y0, Z0 = transformer.transform(xx=actual_origin_proj[0], yy=actual_origin_proj[1], zz=actual_origin_proj[2], direction="FORWARD")

    # Calculate the axial unit vector (based on the actual origin)
    # East direction (x+1 m)
    east_proj = (actual_origin_proj[0] + 1, actual_origin_proj[1], actual_origin_proj[2])
    X_east, Y_east, Z_east = transformer.transform(*east_proj)
    east_vector = np.array([X_east - X0, Y_east - Y0, Z_east - Z0])
    east_unit = east_vector / np.linalg.norm(east_vector)

    # North (y+1 US)
    north_proj = (actual_origin_proj[0], actual_origin_proj[1] + 1, actual_origin_proj[2])
    X_north, Y_north, Z_north = transformer.transform(*north_proj)
    north_vector = np.array([X_north - X0, Y_north - Y0, Z_north - Z0])
    north_unit = north_vector / np.linalg.norm(north_vector)

    # Calculate the upper direction (right-hand rule)
    up_unit = np.cross(east_unit, north_unit)

    # Build a 4x4 transformation matrix (note unit conversion)
    transform = np.eye(4)
    transform[:3, 0] = east_unit * 1.0  # Keep rice units

    transform[:3, 1] = north_unit * 1.0
    transform[:3, 2] = up_unit * 1.0
    transform[:3, 3] = [X0, Y0, Z0]

    # Convert to a one-dimensional array of column main order
    return transform.T.flatten().tolist()


def create_transform_matrix_virtual_origin(lon, lat, alt, scale=None):
    # Convert to epsg:4978 Earth-center coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")
    x0, y0, z0 = transformer.transform(lat, lon, alt)  # Parameter order is latitude, longitude, elevation

    # Calculate the coordinates of ENU base vectors in ECEF
    # Note: the enu2ecef parameter order is east-direction increment, north-direction increment, sky-direction increment, latitude, longitude, elevation
    east_ecef = enu2ecef(1, 0, 0, lat, lon, alt)
    north_ecef = enu2ecef(0, 1, 0, lat, lon, alt)
    up_ecef = enu2ecef(0, 0, 1, lat, lon, alt)

    # Convert to a vector relative to the origin
    east_vector = np.array(east_ecef) - np.array([x0, y0, z0])
    north_vector = np.array(north_ecef) - np.array([x0, y0, z0])
    up_vector = np.array(up_ecef) - np.array([x0, y0, z0])

    # Normalized vector
    east_vector /= np.linalg.norm(east_vector)
    north_vector /= np.linalg.norm(north_vector)
    up_vector /= np.linalg.norm(up_vector)

    # Build a rotation matrix (enu to ecef)
    R = np.eye(4)
    R[:3, 0] = east_vector
    R[:3, 1] = north_vector
    R[:3, 2] = up_vector

    # Building a transformation matrix: Rotate first and then translate
    transform = np.eye(4)
    transform[:3, :3] = R[:3, :3]
    transform[:3, 3] = [x0, y0, z0]

    # Apply zoom (if any)
    if scale:
        S = np.diag([*scale, 1.0])
        transform = S @ transform

    # Transpose to column main order and flatten
    return transform.T.flatten().tolist()


# Generate tileset.json
def generate_tileset_json(output_dir: str, root_node: OctreeNode, bounds: List[float], geometric_error: float, transform_matrix: List[float]):
    def build_tile_structure(node: OctreeNode, tile_name: str, current_geometric_error: float) -> Dict:
        # Check if there is a corresponding GLTF file
        gltf_path = f"{output_dir}/{tile_name}.gltf"
        if not node.children and not os.path.exists(gltf_path):
            return None  # If there is no corresponding GLTF file, return None

        bounding_volume = {"box": compute_box([point.position for point in node.get_all_points()])}
        content = {"uri": f"{tile_name}.gltf"} if not node.children else None
        children = []
        if node.children:
            for i, child in enumerate(node.children):
                child_tile = build_tile_structure(child, f"{tile_name}_{i}", current_geometric_error)
                if child_tile is not None:
                    children.append(child_tile)
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": current_geometric_error,
            "refine": "REPLACE",
        }
        if content:
            tile_structure["content"] = content
        if children:
            tile_structure["children"] = children
        return tile_structure

    root_tile = build_tile_structure(root_node, "tile_0", geometric_error)

    # Add transform to root node
    if transform_matrix is not None:
        root_tile["transform"] = transform_matrix
    tileset = {"asset": {"version": "1.1", "gltfUpAxis": "Z"}, "geometricError": geometric_error, "root": root_tile}

    # Remove empty content from root
    if "content" in tileset["root"] and tileset["root"]["content"] is None:
        del tileset["root"]["content"]

    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f, cls=NumpyEncoder, indent=4)


def compute_box(points: np.ndarray) -> List[float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    center = np.mean(points, axis=0)
    half_size = (np.max(points, axis=0) - np.min(points, axis=0)) / 2
    return [center[0], center[1], center[2], half_size[0], 0, 0, 0, half_size[1], 0, 0, 0, half_size[2]]


def is_crs_wkt(text):
    try:
        CRS.from_wkt(text)
        return True
    except CRSError:
        return False


def get_suffix(filename):
    suffix = filename.rsplit(".", 1)[-1] if "." in filename else ""
    return suffix.lower()  # Convert to lowercase


# Main function
def main(
    input_path: str,
    output_dir: str,
    node_capacity: int,
    geometric_error: float,
    lon: float,
    lat: float,
    alt: float,
    wkt: str,
    wkt_origin: str,
):
    input_type = get_suffix(input_path)

    print("Loading File...")
    if input_type == "splat":
        # Splat file reading stage
        points = read_splat_file(input_path)

    elif input_type == "ply":
        # Ply file reading stage
        points = read_ply_file(input_path)

    else:
        raise ValueError("input file type must be either 'splat' or 'ply'")

    # Build the octree stage
    print("Building Octree...")
    positions = np.array([point.position for point in points], dtype=np.float32)
    points_data = np.array(points, dtype=object)

    # Calculate the boundaries
    min_x, min_y, min_z = np.min(positions, axis=0)
    max_x, max_y, max_z = np.max(positions, axis=0)

    # Create a root node
    root = OctreeNode((min_x, min_y, min_z, max_x, max_y, max_z), capacity=node_capacity)

    # Batch insertion point data
    root.insert_batch(positions, points_data)

    # Print the total number of points converted to 3D Tiles
    total_points_after_conversion = len(root.get_all_points())
    print(f"Convertd to 3D Tiles points num: {total_points_after_conversion}")

    # Generate 3D Tiles Stage
    print("Generating 3dtiles...")
    generate_3dtiles(root, output_dir, "tile_0")

    # Generate coordinate transformation matrix
    if len(wkt) != 0 and is_crs_wkt(wkt):  # With geographic information
        transform_matrix = create_transform_matrix_wkt_origin(wkt, wkt_origin)
    else:  # No geographic information
        transform_matrix = create_transform_matrix_virtual_origin(lon, lat, alt)

    # Generate tileset.json stage
    print("Generating tileset json...")
    bounds = [min_x, min_y, min_z, max_x, max_y, max_z]
    generate_tileset_json(output_dir, root, bounds, geometric_error, transform_matrix)

    print("Done.")
    return True


def run_with_args_list(args_list):
    parser = argparse.ArgumentParser(description="Convert .splat or .ply file to 3D Tiles")
    parser.add_argument("--input_path", type=str, help="Input file path", required=True)
    parser.add_argument("--output_dir", type=str, help="Output 3D Tiles directory", required=True)
    parser.add_argument("--node_capacity", type=int, help="Octree node capacity", default=500000, required=False)
    parser.add_argument("--geometric_error", type=float, help="Input geometric error", default=1, required=False)
    parser.add_argument("--override", action="store_true", help="Whether to override result", default=False, required=False)
    parser.add_argument("--lon", type=float, help="Local origin longitude (EPSG:4326)", default=121.1788904, required=False)
    parser.add_argument("--lat", type=float, help="Local origin latitude (EPSG:4326)", default=31.1597934, required=False)
    parser.add_argument("--alt", type=float, help="Local origin altitude (meters)", default=20, required=False)
    parser.add_argument("--wkt", type=str, help="well-known text for coordinate reference systems", default="", required=False)
    parser.add_argument("--wkt_origin", type=str, help="coordinate reference systems origin xyz", default="", required=False)
    args = parser.parse_args(args_list)

    if args.override:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.listdir(args.output_dir):
        print(f"Warning: The output directory '{args.output_dir}' is not empty. Existing files may interfere with the output results.")

    return main(
        args.input_path,
        args.output_dir,
        args.node_capacity,
        args.geometric_error,
        args.lon,
        args.lat,
        args.alt,
        args.wkt,
        args.wkt_origin,
    )


if __name__ == "__main__":
    run_with_args_list(sys.argv[1:])
