# Pupil 共轭轴向误差 `delta_z` 扫描

这个功能完全离线运行，不连接 SLM、CCD，也不移动 L1、L2、Obj1、Obj2、
tube lens 或 CCD。程序为每个 `delta_z` 使用同一张初始随机相位，独立重新运行
WGS，并输出之后可以逐张加载到 SLM 的 BMP。

## 传播模型

正向模型为：

```text
准直高斯输入场
  -> 1272 x 1024 SLM active-area（外部振幅为零）
  -> 理想 4f SLM 共轭场
  -> Angular Spectrum propagation(delta_z)
  -> Obj1 有限圆形 pupil
  -> Obj1 Fourier transform
  -> target plane
```

反向模型按伴随顺序执行：inverse Fourier transform、同一个 pupil mask、
Angular Spectrum propagation(`-delta_z`)。

有限 pupil 是必要的。如果没有 pupil clipping 或 pupil aberration，数学上有
`abs(FFT(P_delta_z(E))) == abs(FFT(E))`，不同 `delta_z` 只改变 target-plane
相位而不改变强度，扫描不会提供成像质量信息。

## 默认粗扫

在仓库根目录运行：

```bash
python delta_z_scan.py --output-dir delta_z_scan_outputs
```

默认扫描：

```text
-20, -15, -10, -5, 0, 5, 10, 15, 20 mm
```

参数默认来自 `hamamatsu_test_config.json`：波长 0.795 um、SLM 像素
12.5 um、4f 放大率 1。当前 Obj1 为 LD Plan 19X/0.65，焦距 10.5 mm；
按 `f_obj * NA` 得到有效 pupil 半径 6.825 mm，因此默认读取
`maskradius=6825 um`。资料中的 36 mm 是最大通光孔径，不作为有效 pupil
直径使用。这个 pupil 在扫描中始终启用，即使 `delta_z=0` 也启用，保证所有
扫描点使用相同的物镜模型。

SLM 使用 Hamamatsu 的 1272 x 1024 有效 LCOS 像素，像素间距 12.5 um，
有效面积 15.9 x 12.8 mm。控制器传输格式虽然是 1280 x 1024 SXGA，但额外
的 8 列不是有效调制像素。扫描器在 4096 x 4096 padded grid 上对实际
1272 x 1024 矩形以外的输入振幅置零，并将入射波前视为理想准直光。

配置中的阵列间距为 7.875 um。这是把旧的 200 mm 聚焦模型下 150 um 间距
按 `10.5 / 200` 缩放后的结果，因此 8 x 8 阵列在 WGS 计算网格中的位置保持
为原来的 48 pixel 间距。若实验需要其他真实光镊间距，应直接修改配置中的
`spacing`，单位为 Obj1 target/focal plane 上的 um。

可以显式修改 pupil 半径：

```bash
python delta_z_scan.py --pupil-radius-mm 4.5
```

## 细扫

例如在粗扫最优点附近以 0.5 mm 步长扫描：

```bash
python delta_z_scan.py \
  --scan-start-mm -7 \
  --scan-stop-mm -3 \
  --scan-step-mm 0.5 \
  --output-dir delta_z_fine_scan
```

也可以直接给出任意扫描点：

```bash
python delta_z_scan.py --delta-z-mm -2 -1 0 1 2
```

## 输出

每个扫描点生成：

- `slm_phase_scan_###_delta_z_*.npy`：完整计算网格上的相位，单位 rad；
- `slm_phase_scan_###_delta_z_*.bmp`：完整 1272 x 1024 LCOS 区域的 8-bit
  相位图；
- `delta_z_metrics.csv`：仿真指标；
- `metrics_vs_delta_z.png`：指标曲线；
- `scan_parameters.json`：本次扫描的物镜、pupil、SLM、目标阵列、计算网格、
  输入波前模型、BMP 校准状态和随机种子。

BMP 是原始 WGS phase screen；扫描器不会额外叠加 Blink calibration、Zernike
或薄透镜二次相位，因而粗扫中唯一变化的建模参数是 `delta_z`。
`scan_000`、`scan_001` 等前缀保证 USB_SLMControl 按文件名升序批量写入
frame memory 时，slot 顺序与命令行给出的 `delta_z` 顺序一致。默认粗扫即为
`-20, -15, ..., +20 mm`。

每次扫描必须使用新的输出目录。如果目录中已经存在本扫描器生成的 BMP、NPY、
CSV、PNG 或参数 JSON，程序会在计算前报错，避免旧 BMP 混入 USB frame
memory。程序不会自动删除或覆盖这些实验文件。

CSV 指标定义：

- `wgs_error`：目标点归一化强度分布相对期望分布的 L2 误差；
- `target_plane_uniformity`：目标点中的 `min(intensity)/max(intensity)`；
- `background_halo`：目标点邻域外的归一化功率；
- `target_efficiency`：目标像素功率占 pupil 透过功率的比例；
- `pupil_transmission`：相对固定输入功率的 pupil 总透过率；
- `peak_intensity`：相对固定输入功率归一化的全图峰值；
- centroid、peak 位置及二者距离：单位为计算网格 pixel。

这些都是 `sim_*` 性质的收敛与模型指标。硬件接好后，最终最佳 `delta_z`
必须通过固定机械位置、固定曝光/增益/功率下的 CCD 实测图像选择。

## Python 依赖

扫描器沿用仓库现有的 PyTorch、SciPy 和 `Aberration.py`。如果环境尚未安装
依赖，可以运行：

```bash
pip install -r requirements-delta-z.txt
```

4096 x 4096、complex128 的实际运行峰值内存约为 5.3 GiB。若 GPU 显存不足，
应在启动 Python 前通过环境设置禁用 CUDA，让现有 WGS 走 CPU。

## 兼容旧 WGS

原有调用不需要修改。以下调用严格走旧 FFT/IFFT 路径，不计算 Angular
Spectrum，也不增加 pupil mask：

```python
phase = WGS_phase_generate(
    init_amp,
    init_phase,
    target_amp,
    delta_z_mm=0.0,
    pupil_radius_mm=None,
    slm_active_shape=None,
)
```

物理 `delta_z` 扫描应显式提供 `pupil_radius_mm`；命令行扫描器当前根据
LD Plan 19X/0.65 配置默认提供 6.825 mm。
