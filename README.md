# SLM/WGS pupil-conjugation `delta_z` scan

本仓库可以完成两部分工作：

1. 对每一个 pupil 共轭轴向误差 `delta_z` 独立运行 WGS，生成对应的
   `1272×1024`、8-bit SLM phase BMP；
2. 在 Windows 实验电脑上依次把这些 phase 显示到 SLM，用 Allied
   Vision/AVT CCD 拍摄固定的 Obj1 target/focal plane，并根据实拍质量选择
   最佳 `delta_z`。

这里扫描的是传播模型：

```text
ideal 4f SLM image --Angular Spectrum(delta_z)--> actual Obj1 pupil
```

不是移动 L2、Obj1、Obj2、tube lens 或 CCD，也不是给最终 phase 事后增加一个
quadratic defocus。

## 1. 实验连接

- SLM 控制器通过 DVI/HDMI 接到实验电脑显卡，并在 Windows 中设置为
  “扩展这些显示器”。
- SLM 显示传输分辨率设置为 `1280×1024`，Windows 缩放设置为 `100%`。
- Allied Vision 相机连接到同一台电脑，并先用 Vimba X Viewer 确认可以取图。
- Obj2 + tube lens 把 Obj1 target/focal plane 成像到 CCD。
- 开始一轮 `delta_z` 扫描后，L1、L2、Obj1、Obj2、tube lens 和 CCD 全部保持
  固定。可以在扫描前调好 CCD 成像焦点，但不能逐个 `delta_z` 重新对焦。

程序把物理 `1272×1024` phase 放到 `1280×1024` 传输画面的中央：左右各补
4 列，不做插值或横向拉伸。

## 2. Windows 软件安装

建议使用 64-bit Python 3.10 或更高版本。先安装 Allied Vision Vimba X，
包括对应相机的 transport layer/driver。VmbPy 最稳妥的安装方式是使用本机
Vimba X 安装目录中与 SDK 同版本的 `.whl` 文件；只安装 PyPI 包并不等于已经
安装相机驱动和 transport layer。

官方资料：

- [Allied Vision Vimba X 安装](https://docs.alliedvision.com/installation.html)
- [Allied Vision VmbPy Python API 安装](https://docs.alliedvision.com/pythonAPIManual.html)
- [wxPython Windows 安装](https://www.wxpython.org/pages/downloads/)

在 PowerShell 中：

```powershell
git clone https://github.com/Aze-Sh/slm_code.git
cd slm_code

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-delta-z.txt
```

然后安装 Vimba X 自带的 VmbPy wheel。实际路径和版本号以本机安装为准：

```powershell
python -m pip install "C:\path\to\vmbpy-X.Y.Z-py-none-any.whl[numpy]"
```

检查软件接口：

```powershell
python -c "import wx; from vmbpy import VmbSystem; print(wx.version()); print(VmbSystem.get_version())"
python experimental_delta_z_scan.py --help
```

## 3. 生成第一轮 coarse scan

默认扫描：

```text
-20, -15, -10, -5, 0, +5, +10, +15, +20 mm
```

执行：

```powershell
python delta_z_scan.py --output-dir delta_z_scan_outputs
```

这一步不要求连接 SLM 或 CCD。每一个 `delta_z` 都从同一初始 phase 独立运行
WGS，不使用上一点作为 warm start。输出包括：

- `slm_phase_scan_NNN_delta_z_*.bmp`
- `slm_phase_scan_NNN_delta_z_*.npy`
- `delta_z_metrics.csv`
- `metrics_vs_delta_z.png`
- `scan_parameters.json`

## 4. 自动加载 SLM、CCD 拍摄和实拍分析

连接 SLM 与 CCD，关闭可能覆盖 SLM 第二屏的窗口。默认参数复用了原 AVT
notebook 中已经工作过的设置：monitor 1、曝光 `50 µs`、每点 16 帧、加载后
等待 1 秒。

```powershell
python experimental_delta_z_scan.py `
  --scan-dir delta_z_scan_outputs `
  --output-dir delta_z_experiment `
  --monitor 1 `
  --camera-index 0 `
  --exposure-us 50 `
  --frames-per-point 16 `
  --settle-seconds 1
```

程序会按 `scan_000` 到 `scan_008` 的顺序执行：

```text
加载 phase → 等待 SLM/光路稳定 → AVT 连拍 → 帧平均 → 下一 delta_z
```

默认显示前会应用：

- `CAL_LSH0804730_785nm.bmp` factory correction；
- `LUT=224`。

这是仓库中与 795 nm 最接近、且原 Vimba notebook 使用过的校正。如果当前
SLM 序列号不是 `LSH0804730`，不要使用这张 correction，应换成当前设备自己的
校正 BMP；暂时没有对应校正时可加：

```powershell
--no-calibration
```

指定另一张 correction：

```powershell
--correction-bmp C:\calibration\current_slm_795nm.bmp --lut 224
```

### CCD ROI

如果全画面有零级光、反射光或其他亮点，建议只分析光镊阵列所在区域：

```powershell
--roi X Y WIDTH HEIGHT
```

例如：

```powershell
--roi 900 700 700 700
```

程序默认从 `scan_parameters.json` 读取 `8×8=64` 个目标光斑。如果改变了目标
图样，可以显式指定：

```powershell
--expected-spots 64 --min-peak-distance-px 12 --spot-radius-px 8
```

所有 `delta_z` 使用同一组检测到的 CCD 光斑中心和同一个 ROI 半径，保证比较
公平。

### 过曝与原始帧

如果结果提示所有点过曝，请降低 `--exposure-us`，并使用一个新的输出目录重新
扫描。相机输出不是完整 dtype 范围时（例如 12-bit 数据存放在 uint16 中），
可以显式指定饱和值：

```powershell
--saturation-level 4095
```

默认只保存平均图。需要保存每个 `delta_z` 的全部原始帧时加入：

```powershell
--save-raw-frames
```

原始堆栈可能占用数 GB 磁盘空间。

## 5. 实验输出与最佳点判断

`delta_z_experiment` 中包含：

- `camera_average_<scan label>.npy`：浮点平均图；
- `camera_average_<scan label>.tiff`：16-bit 可查看平均图；
- `camera_raw_<scan label>.npy`：可选原始帧；
- `experimental_metrics.csv`：所有实拍指标；
- `experimental_metrics_vs_delta_z.png`：实拍指标曲线；
- `detected_spots.png`：所有扫描点共用的光斑 ROI；
- `best_delta_z.json`：推荐的 coarse-scan 最佳点；
- `experimental_parameters.json`：曝光、帧数、SLM 分辨率、校正、ROI 和来源
  WGS 参数。

主要实拍指标：

- `target_plane_uniformity`：64 个光斑积分强度的 `min/max`，越高越好；
- `spot_intensity_cv`：光斑积分强度的 `std/mean`，越低越好；
- `background_halo`：固定光斑 ROI 外的能量比例，越低越好；
- `mean_fwhm_px`：平均等效 FWHM，越低越清晰；
- `mean_encircled_energy_radius_50_px`：50% 包围能量半径，越低越集中；
- `mean_spot_sharpness`：光斑能量集中度，越高越好；
- `saturation_fraction`：过曝像素比例；
- centroid、peak 及其偏移。

`quality_score` 对以下四项做本轮扫描内的归一化并等权平均：均匀性、sharpness、
反向 halo、反向 FWHM。它用于快速排序；最终选择时仍应同时查看原始 CCD 图、
`detected_spots.png` 和各条独立指标。过曝点不会被自动推荐。

## 6. 最佳区域的 fine scan

假设 coarse scan 的最佳点是 `+5 mm`，可以进一步扫描 `+2` 到 `+8 mm`，步长
`0.5 mm`：

```powershell
python delta_z_scan.py `
  --output-dir delta_z_fine_outputs `
  --scan-start-mm 2 `
  --scan-stop-mm 8 `
  --scan-step-mm 0.5

python experimental_delta_z_scan.py `
  --scan-dir delta_z_fine_outputs `
  --output-dir delta_z_fine_experiment `
  --monitor 1 `
  --exposure-us 50 `
  --frames-per-point 16 `
  --settle-seconds 1
```

coarse scan 和 fine scan 应保持同一套 CCD ROI、曝光、平均帧数、校正方式以及
全部机械位置。

## 7. 常见问题

### `Invalid monitor`

Windows 没有识别到对应编号的扩展显示器。检查显示设置，或尝试
`--monitor 2`。不要使用“复制这些显示器”。

### 检测到的光斑数不足

先查看各个 CCD TIFF，确认没有过曝或完全无光；然后缩小 `--roi`、减小
`--min-peak-distance-px`，或核对 `--expected-spots`。

### `Invalid VmbC Version`

VmbPy wheel 与 Vimba X SDK 版本不匹配。重新安装当前 Vimba X 安装目录中附带
的 VmbPy wheel。

### 输出目录已存在

程序不会覆盖或混合旧实验结果。换一个新的 `--output-dir`。
