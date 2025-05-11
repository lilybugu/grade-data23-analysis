@echo off
echo 正在创建Python虚拟环境...
python -m venv venv

echo 激活虚拟环境...
call venv\Scripts\activate.bat

echo 安装所需的Python包...
pip install -r requirements.txt

echo 环境安装完成！
echo 请使用 "venv\Scripts\activate.bat" 激活环境后再运行Python脚本。
pause 