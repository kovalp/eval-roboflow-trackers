from argparse import ArgumentParser
from typing import Sequence

from rich_argparse import RawTextRichHelpFormatter


PROG = 'eval-bytetrack-clavia'


class ClavIaEvalCmdLine:
    def __init__(self):
        self.verbosity: int = 0
        self.glob: str = 'data/mot17/val/MOT17-*-FRCNN/gt/gt.txt'

    def __repr__(self) -> str:
        return f'ClavIaEvalCmdLine({vars(self)})'


def get_cmd_line(args: Sequence[str]) -> ClavIaEvalCmdLine:
    cli = ClavIaEvalCmdLine()
    parser = ArgumentParser(PROG, f'{PROG} [OPTIONS]', formatter_class=RawTextRichHelpFormatter)
    parser.add_argument('--glob', '-g', default=cli.glob, help='MOT17 ground-truth')
    parser.add_argument('--verbosity', '-v', action='count', help='Verbosity')
    parser.parse_args(args, namespace=cli)
    if cli.verbosity > 0:
        print(cli)
    return cli
