# ~/work/vedit/calflow/setup.py

from setuptools import setup, find_packages

setup(
    name="calflow",
    version="0.1.0",

    # --- 核心修正 1 ---
    # 我们不再使用 find_packages() 来寻找主包。
    # 我们在这里显式声明我们要创建和安装的“顶级包”有哪些。
    # 最重要的是，我们声明了一个名为 'calflow' 的包。
    packages=[
        'calflow',
        'gmflow',
        # 'raft' # 根据您的 ls -R 输出，deps/RAFT 为空，暂时注释掉。如果该目录下有带 __init__.py 的 core 目录，请取消注释。
    ],

    # --- 核心修正 2 ---
    # 这是实现您目标的关键。这个字典告诉 setuptools 在哪里找到上面 'packages' 列表中声明的包的源代码。
    package_dir={
        # 关键映射：告诉 setuptools，“calflow”这个包的源代码就在根目录 ('.')。
        # 这样一来，根目录下的 __init__.py, main.py, utils/, motion_compensation/ 都会被打包进 calflow 命名空间。
        'calflow': '.',

        # 您之前的配置是正确的，我们保留它，用于处理依赖。
        # 包名 'gmflow' -> 它的源代码在 'deps/gmflow/gmflow' 目录
        'gmflow': 'deps/gmflow/gmflow',

        # 'raft': 'deps/RAFT/core', # 同样，如果需要，请取消注释。
    },

    # 其他配置保持不变
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "matplotlib",
        "imageio",
        "pandas",
        "imageio-ffmpeg",
    ],
    entry_points={
        'console_scripts': [
            # 注意：因为 main.py 现在是 calflow 包的一部分，所以入口点需要更新
            'calflow=calflow.main:main',
        ],
    },
)