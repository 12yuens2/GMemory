"""The sweep entry point: `python tasks/run.py --task fever --mas_type autogen ...`

The flags themselves are `sweep.build_arg_parser`, and one experiment of a sweep
is `experiment.py`.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sweep import main

if __name__ == '__main__':
    main(sys.argv[1:])
