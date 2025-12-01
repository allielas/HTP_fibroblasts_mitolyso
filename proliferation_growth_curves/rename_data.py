import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Union
from openpyxl import load_workbook

#!/usr/bin/env python3
"""
rename_data.py

Load an Excel file, find & replace strings in cells based on a list of replacements,
and write the result to a new Excel file (or overwrite).

Usage examples:
    python rename_data.py -i data.xlsx -m mappings.csv
    python rename_data.py -i data.xlsx -m mappings.json -o data_renamed.xlsx
    python rename_data.py -i data.xlsx --map "old:new" --map "foo:bar" --inplace

Mappings:
    - A CSV with two columns (old,new) or (source,target)
    - One or more --map "old:new" pairs on the command line
"""


def load_mappings(path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mappings file not found: {path}")
    else:
        # assume CSV/TSV: first two columns are old,new
        df = pd.read_csv(p, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("CSV mappings must have at least two columns (old, new).")
        old_col, new_col = df.columns[0], df.columns[1]
        return {
            str(o): str(n)
            for o, n in zip(df[old_col].fillna(""), df[new_col].fillna(""))
        }


def parse_inline_maps(map_args: List[str]):
    mappings = {}
    for s in map_args:
        if ":" not in s:
            raise ValueError(f"Inline mapping must be OLD:NEW, got: {s!r}")
        old, new = s.split(":", 1)
        mappings[old] = new
    return mappings


def apply_replacements(df: pd.DataFrame, mapping: Dict[str, str]):
    if not mapping:
        return df.copy()
    # For performance: compile mapping items once
    items = list(mapping.items())

    def replace_value(x):
        """
        Docstring for replace_value. Just a simple string replace

        :str x: the string to replace based on the dict
        """
        if isinstance(x, str):
            for old, new in items:
                if old:
                    x = x.replace(old, new)
            return x
        return x

    return df.applymap(replace_value)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Find & replace strings in an Excel file based on mappings."
    )
    p.add_argument("-i", "--input", required=True, help="Input Excel file (.xlsx/.xls)")
    p.add_argument(
        "-s",
        "--sheet_name",
        default=None,
        help="Sheet name or index (passed to pandas.read_excel).",
    )
    p.add_argument("-m", "--mappings", help="Path to mappings file (CSV or JSON).")
    p.add_argument(
        "--map",
        action="append",
        default=[],
        help='Inline mapping "OLD col : NEW col". Can be repeated.',
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output Excel file. If omitted, will add _renamed before suffix.",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file (mutually exclusive with --output).",
    )
    args = p.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file does not exist: {inp}", file=sys.stderr)
        sys.exit(2)

    if args.inplace and args.output:
        print("Cannot use --inplace and --output together.", file=sys.stderr)
        sys.exit(2)

    # load mappings dict from csv
    mapping: Dict[str, str] = {}
    if args.mappings:
        mapping.update(load_mappings(args.mappings))
    if args.map:
        mapping.update(parse_inline_maps(args.map))

    # read excel from pandas
    read_kwargs = {}
    if args.sheet_name is not None:
        # allow numeric index if user provides digits
        try:
            sheet_name = int(args.sheet_name)
            read_kwargs["sheet_name"] = sheet_name
            sheet_name = str(sheet_name)
        except ValueError:
            sheet_name = args.sheet_name
            read_kwargs["sheet_name"] = sheet_name
    else:
        sheet_name = "Sheet1"

    df = pd.read_excel(inp, **read_kwargs)

    # apply replacements
    out_df = apply_replacements(df, mapping)

    # determine output path
    if args.inplace:
        out_path = inp
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = inp.with_name(inp.stem + "_renamed" + inp.suffix)

    if Path.exists(out_path):
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="a") as writer:
                out_df.to_excel(writer, sheet_name=sheet_name)
        except ValueError:
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
                out_df.to_excel(writer, sheet_name=sheet_name)

    else:
        # write back to excel
        out_df.to_excel(out_path, index=False)
    print(f"Wrote replaced data to: {out_path}")



ex1 = "0_2024-01-25"
ex2 = "0_2024-03-01"


for i in range(5,8):
    plate = i
    args = [
        "-i",
        f"/Users/allielas/HTP_fibroblasts_mitolyso/proliferation_growth_curves/Incucyte_sheets/r{plate}.xlsx",
        "-m",
        f"/Volumes/AllieS/Incucyte_Data_new/full_data/excel sheets/replacements/plate{plate}_replacements.csv",
        "-o",
        f"/Users/allielas/HTP_fibroblasts_mitolyso/proliferation_growth_curves/Incucyte_sheets/renamed/r{plate}_replaced.xlsx",
        "-s",
        "2",
    ]
    if __name__ == "__main__":
        main(args)
