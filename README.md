# 集成隐写/分析平台
# 安装教程(Linux)
1. 下载SDK

https://github.com/h2oai/wave/releases


(使用教程:https://wave.h2o.ai/docs/getting-started)

2. 解压并移动到home目录

`tar -xzf wave-0.15.0-linux-amd64.tar.gz`

`mv wave-0.15.0-linux-amd64 $HOME/wave`

3. 运行服务端进程

在`$HOME/wave`目录下执行`./waved`

4. 运行app进程


`git clone https://github.com/kevin2li/integrated_stega_paltform.git`

`cd integrated_stega_paltform/web`

`wave run index`

# 效果展示
![](assets/image_steganalysis.png)
