"""Command line interface for orgatools.

A single ``orgatools`` entry point, with the data tools under ``orgatools data``. Command callbacks import their
dependencies lazily, so ``orgatools --help`` stays fast.
"""

import os

import click


CLI_EPILOG = """\
Examples:

\b
  # Convert a single tif file to OME-Zarr
  orgatools data convert -i stack.tif -o output/

\b
  # Convert every tif in a folder (and its subfolders) to OME-Zarr
  orgatools data convert -i stacks/ -o output/
"""


@click.group(context_settings=dict(help_option_names=["-h", "--help"]), epilog=CLI_EPILOG)
@click.version_option(package_name="orgatools", prog_name="orgatools")
def cli():
    """Tools for the organoid data of the Beghin lab."""


@cli.group("data")
def data_group():
    """Data handling: conversion of the microscopy data."""


def _folder_size(path):
    return sum(os.path.getsize(os.path.join(root, name)) for root, _, names in os.walk(path) for name in names)


@data_group.command("convert", epilog=CLI_EPILOG)
@click.option(
    "-i", "--input", "input_path", required=True, type=click.Path(exists=True),
    help="A tif file or a folder with tif files.",
)
@click.option(
    "-o", "--output", "output_folder", required=True, type=click.Path(file_okay=False),
    help="The folder for the OME-Zarr files.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite OME-Zarr files that exist already.")
def data_convert(input_path, output_folder, overwrite):
    """Convert tif stacks to OME-Zarr, lossless and checked against the tif after writing.

    A folder is searched recursively and its layout is kept in the output folder.
    """
    from rich.console import Console
    from rich.progress import track
    from rich.table import Table

    from .conversion import convert_tif, find_tifs

    console = Console()
    paths = find_tifs(input_path, output_folder)
    if not paths:
        raise click.ClickException(f"No tif files found in {input_path}.")

    table = Table("Input", "Axes", "Shape", "Voxel size [um]", "tif -> OME-Zarr [MB]")
    table.columns[0].overflow = "fold"
    for tif_path, output_path in track(paths, description="Converting", console=console):
        if os.path.exists(output_path) and not overwrite:
            console.print(f"[yellow]Skipping {output_path}, it exists already.[/yellow]")
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shape, axes, voxel_size = convert_tif(tif_path, output_path)
        table.add_row(
            os.path.relpath(tif_path, input_path) if os.path.isdir(input_path) else os.path.basename(tif_path),
            axes, "x".join(str(size) for size in shape), "x".join(f"{size:.3g}" for size in voxel_size.values()),
            f"{os.path.getsize(tif_path) / 1e6:.1f} -> {_folder_size(output_path) / 1e6:.1f}",
        )

    if table.row_count:
        console.print(table)
    console.print(f"[green]Converted {table.row_count} of {len(paths)} tif files to {output_folder}.[/green]")
