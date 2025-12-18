#!/bin/bash

CMD="python perl/eval/monitor.py --result-dir outputs/eval --serve --port 5000"

while true; do
    $CMD
    RET=$?

    echo ">>> 程序退出，返回码：$RET"
    echo ">>> 5 秒后重新启动..."
    sleep 5
done