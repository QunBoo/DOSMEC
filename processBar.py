import time
import sys


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    sys.stdout.write('\r|%s| %s%%' % (bar, int(percent)))
    sys.stdout.flush()


if __name__ == '__main__':
    for i in range(100):
        time.sleep(0.1)
        progress_bar(i, 100)
    print()