'''此程序为自动化运行脚本，方便自动修改参数以训练网络'''

import datetime
import os
import threading
import time


def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))


if __name__ == '__main__':

    print('开始时间：{}'.format(time.strftime('%m-%d_%H%M')))
    start_time = time.time()

    # 是否需要并行运行
    if_parallel = False   # 显存较小，串行比较合适

    # 需要执行的命令列表，按需求修改即可
    cmds = [
        'python train.py --task=W --weather=mix',
        'python train.py --task=P --weather=mix',
        'python train.py --task=P --weather=foggy',
        'python train.py --task=P --weather=rainy',
        'python train.py --task=P --weather=snowy',
        'python train.py --task=W --weather=mix --model=mynet',
        'python train.py --task=P --weather=mix --model=mynet',
        'python train.py --task=P --weather=foggy --model=mynet',
        'python train.py --task=P --weather=rainy --model=mynet',
        'python train.py --task=P --weather=snowy --model=mynet',
    ]

    if if_parallel:
        # 并行
        threads = []
        for cmd in cmds:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)

        # 等待线程运行完毕
        for th in threads:
            th.join()
    else:
        # 串行
        for cmd in cmds:
            try:
                print("命令%s 开始运行: %s" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
                os.system(cmd)
                print("命令%s 结束运行: %s\n" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
            except:
                print('%s\t 运行失败' % (cmd))

    end_time = time.time()
    total_time = end_time - start_time
    print('总用时: {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))

