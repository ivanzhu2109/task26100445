# synthocc — 单场景图像合成与逐实例遮挡标注库

为计算机视觉训练数据生成而设计：调用方提供场景描述（画布、背景、
从后到前的实例及其仿射变换），本库输出合成 RGB PNG 与逐实例 JSON 标注。

## 环境

- Python ≥ 3.10，依赖 `numpy` 与 `Pillow`（测试另需 `pytest`）。
- 纯本地文件读写，不启动任何网络/端口服务。

## 快速开始

```python
from pathlib import Path
from synthocc import Affine2D, Instance, Scene, render_scene

scene = Scene(
    width=640,
    height=480,
    background=(24, 24, 24),                 # 不透明纯色背景
    instances=(
        Instance("inst-001", "cat", Path("assets/cat.png"),
                 Affine2D((1.0, 0.0, 100.0, 0.0, 1.0, 80.0))),   # 平移
        Instance("inst-002", "cat", Path("assets/cat.png"),
                 Affine2D((0.866, -0.5, 300.0, 0.5, 0.866, 120.0))),  # 旋转30°
    ),
)
result = render_scene(scene, Path("out/"), stem="scene_0001")
# result.image_path      -> out/scene_0001.png   (RGB PNG)
# result.annotation_path -> out/scene_0001.json  (逐实例标注)
# result.annotations     -> 内存中的标注元组
```

- 资产为本地 RGBA PNG；每个实例给出**源坐标 → 画布坐标**的可逆二维仿射
  变换 `(a, b, c, d, e, f)`，即 `canvas = [[a,b,c],[d,e,f]] @ [sx, sy, 1]`。
- 实例元组按**从后到前**排列，后者覆盖前者。
- 相同资产或类别的不同实例各自独立标注，不会合并。

## 输出 JSON 格式

```json
{
  "width": 640,
  "height": 480,
  "background": [24, 24, 24],
  "instances": [
    {"instance_id": "inst-001", "category": "cat",
     "occlusion": 12.34, "bbox": [x0, y0, x1, y1]},
    {"instance_id": "inst-002", "category": "cat",
     "occlusion": 100.0, "bbox": null}
  ]
}
```

- `bbox` 为包住全部正可见权重像素的最小整数矩形，右/下边界**不包含**；
  完全遮挡的实例保留标注，`occlusion` 为 100，`bbox` 为 `null`。
- 画布内自身采样透明度总和为零的实例不产生标注。
- `occlusion` = 100 × (1 − 可见权重总和 / 自身采样透明度总和)，
  两者均只统计画布内像素：裁出画布的部分**不算**遮挡，同色覆盖仍算遮挡。

## 数值约定

- 源图与画布均以左上角为原点，像素中心坐标为整数 + 0.5。
- 画布像素按其中心映射到源图的位置取样，采用**预乘透明度**的双线性插值；
  源图外视为透明黑，全透明像素下隐藏的 RGB 不影响成图。
- 透明度按 0–1 解释；颜色直接在 8 位 RGB 尺度上以浮点合成，
  **仅最终输出时**四舍五入到整数，中间层不量化。

## 运行测试

```bash
python3 -m pytest tests/ -q
```

测试包含一个纯 Python 逐像素的独立参考实现（`tests/reference.py`），
对旋转、缩放、亚像素平移、画布裁剪、透明孔洞、半透明边缘、
全透明但 RGB 非零像素、同类别重叠、多前景覆盖同一区域、
完全遮挡等场景，交叉校验图像、边界框与遮挡率。

## 代码结构

- `synthocc/types.py` — 场景/实例/标注等数据类型与仿射逆变换
- `synthocc/sampler.py` — 预乘透明度双线性采样
- `synthocc/compositor.py` — 从后到前透明合成（浮点，不中间量化）
- `synthocc/annotation.py` — 可见权重、遮挡率与可见边界框
- `synthocc/api.py` — `render_scene` 顶层入口（读资产、写 PNG/JSON）
