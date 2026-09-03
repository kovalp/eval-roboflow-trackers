from argparse import ArgumentParser
from typing import Sequence

from rich_argparse import RawTextRichHelpFormatter


PROG = 'eval-bytetrack-clear-detections'


class ClearDetEvalCmdLine:
    def __init__(self):
        self.verbosity: int = 0
        self.glob_det: str = 'data/mot17/val/MOT17-*-FRCNN/det/det.txt'
        self.glob_gt: str = 'data/mot17/val/MOT17-*-FRCNN/gt/gt.txt'

    def __repr__(self) -> str:
        return f'ClearDetEvalCmdLine({vars(self)})'


def get_cmd_line(args: Sequence[str]) -> ClearDetEvalCmdLine:
    cli = ClearDetEvalCmdLine()
    parser = ArgumentParser(PROG, f'{PROG} [OPTIONS]', formatter_class=RawTextRichHelpFormatter)
    parser.add_argument('--glob-det', '-d', default=cli.glob_det, help='MOT17 detections')
    parser.add_argument('--glob-gt', '-g', default=cli.glob_gt, help='MOT17 ground truth')
    parser.add_argument('--verbosity', '-v', action='count', help='Verbosity')
    parser.parse_args(args, namespace=cli)
    if cli.verbosity > 0:
        print(cli)
    return cli
