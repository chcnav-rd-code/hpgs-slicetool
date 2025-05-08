
# HPGS Slice Tool

![version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![license](https://img.shields.io/badge/license-Apache-green.svg)
![python](https://img.shields.io/badge/python-3.8%2B-yellow.svg)

> This project provides a lightweight and efficient tool for converting 3D Gaussian Splatting (3DGS) data from .splat or .ply format to the 3DTiles .gltf format. 

---

## Installation

```bash
# Clone the repository
git clone https://github.com/CHCNAV-Official/HPGS.git
cd ./HPGS/slicetool

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Command-line

```bash
python gs_to_gltf.py \
  --input_path path/to/input.ply \
  --output_dir folder/to/output
```

> ⚠️ **Note:** 
> 3DTiles require all coordinates (`xyz`) to be in the Earth-Centered, Earth-Fixed (ECEF) system (EPSG:4978).  
>   
> - If your 3DGS data **does not have geographic reference**, please provide a **virtual local origin** using geographic coordinates via `--lon`, `--lat`, and `--alt`. *(By default, the virtual origin is set at Huace Company in Shanghai.)*  
>   
> - If your 3DGS data **is already in a projected coordinate system**, please provide the corresponding `--wkt` string and use `--wkt_origin` to specify the offset of the 3DGS data. *(In this case, the 3DTiles should align accurately with the Cesium basemap.)*

### Parameters
| Parameter           | Type    | Description                                                                   |
| ------------------- | ------- | ----------------------------------------------------------------------------- |
| `--input_path`      | `str`   | **(Required)** Path to the input `.ply` or `.splat` file.                     |
| `--output_dir`      | `str`   | **(Required)** Path to the output directory where result will be saved.     |
| `--node_capacity`   | `int`   | Maximum number of points per octree node. Default is `500000`.                |
| `--geometric_error` | `float` | Base geometric error. Default is `1.0`.   |
| `--override`        | `flag`  | If set, existing results in the output directory will be overwritten.         |
| `--lon`             | `float` | Longitude of the virtual origin (EPSG:4326). Default is `121.1788904`.          |
| `--lat`             | `float` | Latitude of the virtual origin (EPSG:4326). Default is `31.1597934`.            |
| `--alt`             | `float` | Altitude of the virtual origin in meters. Default is `10`.                      |
| `--wkt`             | `str`   | WKT (Well-Known Text) string to define the coordinate system. |
| `--wkt_origin`      | `str`   | Real origin `x, y, z` in the projected coordinate system. |

---

## Viewer

You can visualize the output 3DTiles using a customized Cesium viewer with splatting support.

### 🔧 Recommended Viewer

Use the Cesium [splat-shader](https://github.com/CesiumGS/cesium/tree/splat-shader) branch. A compatible version is included in this repository: `cesium_viewer.7z` .

### ▶️ How to launch

1. Place your 3DTiles data into the `cesium_viewer/data/3dtiles/` folder.  
2. Start a local HTTP server:

    ```bash
    cd cesium_viewer
    python -m http.server 8000
    ```

3. Open your browser and visit:
   [http://localhost:8000/](http://localhost:8000/)

### 🕹️ Navigation

The viewer currently supports **first-person view only**, similar to the `FPS` mode in `SIBR_Viewer`.

#### Movement

| Key | Action        |
|-----|---------------|
| `W` | Move Forward  |
| `S` | Move Backward |
| `A` | Strafe Left   |
| `D` | Strafe Right  |
| `Q` | Move Down     |
| `E` | Move Up       |

#### Rotation

| Key | Action        |
|-----|---------------|
| `J` | Yaw Left      |
| `L` | Yaw Right     |
| `I` | Pitch Up      |
| `K` | Pitch Down    |
| `U` | Roll Left     |
| `O` | Roll Right    |

Mouse movement is not supported for rotation; use keyboard keys for full camera control.

### ⚠️ Tips

- Browser caching may cause rendering issues or outdated data to appear. 
  👉 **Please disable network caching in your browser’s developer tools (F12 > Network > "Disable cache")** during development or repeated viewing.

---

## Known Issues

- The current 3D Tiles extension only supports **zero-degree spherical harmonics**.
  See reference: [KHR_gaussian_splatting](https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting#limitations)

- Cesium currently exhibits **incorrect occlusion between tiles**, due to each tile being **sorted independently** rather than globally.
  See reference: [cesium_occlusion_issue](https://github.com/CesiumGS/cesium/issues/12590)

---

## License

This project is licensed under the Apache License
