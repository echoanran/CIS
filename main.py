#!/usr/bin/env python
import argparse
import sys
# import pdb

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['train-image-causal'] = import_class('processor.train-image-causal.REC_Processor')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    
    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    # pdb.set_trace()

    p.start()
