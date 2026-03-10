"""brane - Local OCR for text and sheet music."""

import click


class DefaultGroup(click.Group):
    """Click group that defaults to 'ocr' when no subcommand is given."""

    def __init__(self, *args, default_cmd="ocr", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd = default_cmd

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [self.default_cmd] + args
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup)
def main():
    """brane - Local OCR for text and sheet music."""


from music_engine import music  # noqa: E402
from ocr_engine import ocr  # noqa: E402

main.add_command(ocr)
main.add_command(music)
