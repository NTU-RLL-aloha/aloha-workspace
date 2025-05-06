import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aloha/interbotix_ws/src/aloha/scripts/install/my-test-package'
