# task26100445

`occlugen`：单场景图像合成与逐实例遮挡标注库。上层生成任务提供场景描述
（画布、背景、实例摆放），库负责透明合成、可见权重与遮挡率计算，并把
RGB PNG 与逐实例 JSON 标注写入调用方指定目录。

## 安装与依赖

纯 Python 包，无服务、无端口、无数据库。依赖：`numpy`、`Pillow`。
直接在仓库根目录 `import occlugen` 即可（或把根目录加入 `PYTHONPATH`）。

## 调用方式

```python
from occlugen import AffineTransform, InstanceSpec, SceneSpec, render_to_dir

scene = SceneSpec(
    width=640,
    height=480,
    background=(255, 255, 255),          # 不透明纯色背景
    instances=(                          # 从后到前排列
        InstanceSpec(
            instance_id="inst-001",      # 实例唯一标识（同类别也要区分）
            category="box",
            asset_path="/path/to/asset_rgba.png",   # 本地 RGBA PNG
            transform=AffineTransform(   # 源坐标 -> 画布坐标的可逆仿射
                a=1.2, b=-0.3, c=100.0,  # canvas_x = a*sx + b*sy + c
                d=0.3, e=1.2,  f=50.0,   # canvas_y = d*sx + e*sy + f
            ),
        ),
        # ...更多实例
    ),
)

annotations = render_to_dir(scene, "/path/to/output_dir")
```

输出目录内容：

- `image.png`：合成后的 RGB PNG（全程浮点计算，仅最终四舍五入量化）。
- `<instance_id>.json`：每个实例一个标注文件，例如：

```json
{
  "instance_id": "inst-001",
  "category": "box",
  "bbox": {"x": 100, "y": 50, "width": 73, "height": 41},
  "occlusion_rate": 12.5
}
```

标注语义：

- `bbox`：包住全部正可见权重像素的最小整数矩形，右/下边界不包含；
  完全遮挡的实例保留标注，`bbox` 为 `null`，`occlusion_rate` 为 `100`。
- `occlusion_rate`：`100 * (1 - 可见权重总和 / 自身采样透明度总和)`，
  两个总和只统计画布内像素，裁出画布的部分不算遮挡，同色覆盖也算遮挡。
- 画布内自身采样透明度总和为零的实例（如完全移出画布）不产生标注。

如需内存结果而不落盘，可用 `render_scene(scene, load_asset)`，返回
`(uint8 RGB ndarray, List[InstanceAnnotation])`。

## 坐标与采样约定

- 源图与画布均以左上角为原点，像素中心坐标为整数加 0.5。
- 画布像素按其中心对应的源位置取样，预乘透明度双线性插值，源图外为
  透明黑；全透明像素隐藏的 RGB 不影响成图。
- 透明度按 0..1 解释，颜色在 8 位 RGB 尺度上直接计算，不做色彩空间转换。

## 测试

```bash
python3 -m pytest tests/ -q
```

测试包含一个独立的纯 Python 逐像素参考实现（`tests/reference.py`），
覆盖旋转/缩放/亚像素平移/画布裁剪、透明孔洞、半透明边缘、隐藏 RGB、
同类别实例重叠、多前景覆盖同一区域、完全遮挡与零透明度实例等场景。
