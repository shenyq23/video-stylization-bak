# 运动向量/光流运动补偿测试

## 环境配置

使用 Anaconda 管理环境，环境导出在根目录的 `environment.yaml`。使用下述命令创建 Anaconda 环境：

```bash
conda env create -f ./environment.yaml
conda activate gmflow
```

所依赖的 GMFlow 模型是仓库的 submodule，所需要的 pretrain 模型上传到了下述链接：

https://drive.google.com/file/d/1yHdW9nEsu03XDnsAnAX6mjL4wQ3cQbvl/view?usp=sharing

下载后解压放到 `/deps/gmflow/pretrained`。

libx265 使用快手内部修改版本，编译完成后的二进制放到 `/bin` 下。

需要跑的视频放到 `/motion_compensation/input/<video-name>` 下，该目录下放一个原视频命名为 `input.mp4`，一个风格化后的视频命名为 `stylized.mp4`。

## 运行命令

`python main.py --video_name <video-name> --batch_size <batch-size> --flow_model [gmflow|x265|mix|reverse_mix]`

其余参数参考 `main.py` 的 `argparser` 配置，batch size 推荐设置为 `4`（V1）或者 `16`（V2）。

运行后的视频存储在 `/motion_compensation/output/<video-name>`，包含一个像素空间的比较（从左到右是风格化、补偿视频、原视频），一个 flow 和 occlusion 的可视化和一个 occlusion 占比走势图。