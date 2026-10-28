"""
Interactive data cleaning & filtering tool.

Run:  python interactive_cleaner.py [optional_path_to_csv_or_xlsx]
Needs: pandas (and openpyxl for .xlsx files)
"""
import sys
import numpy as np
import pandas as pd


# ---------- small input helpers ----------
def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def choose(prompt, options):
    """Show a numbered list, return the chosen option (or None to cancel)."""
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        v = ask(f"{prompt} (number, or Enter to cancel)")
        if not v:
            return None
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1]
        print("  Invalid choice.")


def pick_columns(df, prompt="Column name(s), comma-separated"):
    raw = ask(prompt)
    if not raw:
        return []
    cols = [c.strip() for c in raw.split(",")]
    bad = [c for c in cols if c not in df.columns]
    if bad:
        print(f"  Unknown column(s): {bad}")
        return []
    return cols


def numeric_cols(df):
    return df.select_dtypes("number").columns.tolist()


# ---------- state with undo + action log ----------
class Session:
    def __init__(self, df):
        self.df = df
        self.history = []
        self.log = []

    def apply(self, new_df, note):
        self.history.append(self.df)
        self.df = new_df
        self.log.append(note)
        print(f"  Done: {note}  -> now {self.df.shape[0]} rows x {self.df.shape[1]} cols")

    def undo(self):
        if not self.history:
            print("  Nothing to undo.")
            return
        self.df = self.history.pop()
        print(f"  Undid: {self.log.pop()}  -> now {self.df.shape[0]} rows x {self.df.shape[1]} cols")


# ---------- actions ----------
def profile(s):
    df = s.df
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} cols | duplicates: {df.duplicated().sum()}")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    print(info.to_string())
    num = numeric_cols(df)
    if num:
        print("\nNumeric summary:")
        print(df[num].describe().T[["min", "mean", "50%", "max", "std"]].round(3).to_string())


def preview(s):
    n = int(ask("Rows to show", 10))
    print(s.df.head(n).to_string())


def drop_duplicates(s):
    subset = pick_columns(s.df, "Columns that define a duplicate (Enter = whole row)") or None
    new = s.df.drop_duplicates(subset=subset)
    s.apply(new, f"drop_duplicates(subset={subset}) removed {len(s.df) - len(new)} rows")


def handle_missing(s):
    df = s.df
    miss = df.isna().sum()
    print(miss[miss > 0].to_string() if (miss > 0).any() else "  No missing values.")
    if not (miss > 0).any():
        return
    action = choose("Action", [
        "Drop rows with any missing value",
        "Drop rows missing in specific columns",
        "Drop columns with more than X% missing",
        "Fill numeric columns (mean / median / constant / interpolate)",
        "Fill with most frequent value (mode)",
        "Forward/backward fill",
    ])
    if action is None:
        return
    if action.startswith("Drop rows with any"):
        s.apply(df.dropna(), "dropna(any)")
    elif action.startswith("Drop rows missing in"):
        cols = pick_columns(df)
        if cols:
            s.apply(df.dropna(subset=cols), f"dropna(subset={cols})")
    elif action.startswith("Drop columns"):
        thr = float(ask("Threshold %", 50))
        drop = df.columns[df.isna().mean() * 100 > thr].tolist()
        s.apply(df.drop(columns=drop), f"dropped columns >{thr}% missing: {drop}")
    elif action.startswith("Fill numeric"):
        cols = pick_columns(df, "Numeric column(s) (Enter = all numeric)") or numeric_cols(df)
        method = choose("Method", ["mean", "median", "constant", "interpolate"])
        if method is None:
            return
        group = ask("Compute within groups of column (e.g. country)? Enter = no")
        if group and group not in df.columns:
            print("  Unknown group column.")
            return
        new = df.copy()
        for c in cols:
            if method == "constant":
                continue
            if method == "interpolate":
                f = (lambda x: x.interpolate(limit_area="inside"))
            else:
                f = (lambda x, m=method: x.fillna(getattr(x, m)()))
            new[c] = new.groupby(group)[c].transform(f) if group else f(new[c])
        if method == "constant":
            val = float(ask("Constant value", 0))
            new[cols] = new[cols].fillna(val)
        s.apply(new, f"fill {cols} using {method}" + (f" per {group}" if group else ""))
    elif action.startswith("Fill with most"):
        cols = pick_columns(df)
        new = df.copy()
        for c in cols:
            if new[c].mode().size:
                new[c] = new[c].fillna(new[c].mode().iloc[0])
        s.apply(new, f"fill mode {cols}")
    elif action.startswith("Forward"):
        cols = pick_columns(df, "Column(s) (Enter = all)") or df.columns.tolist()
        limit = int(ask("Max consecutive fills", 3))
        group = ask("Within groups of column? Enter = no")
        new = df.copy()
        for c in cols:
            if group and group in df.columns:
                new[c] = new.groupby(group)[c].transform(lambda x: x.ffill(limit=limit).bfill(limit=limit))
            else:
                new[c] = new[c].ffill(limit=limit).bfill(limit=limit)
        s.apply(new, f"ffill/bfill {cols} limit={limit}" + (f" per {group}" if group else ""))


def clean_text(s):
    df = s.df
    cols = pick_columns(df, "Text column(s) (Enter = all text columns)") or df.select_dtypes(["object", "string"]).columns.tolist()
    if not cols:
        print("  No text columns.")
        return
    case = choose("Case", ["leave as is", "lowercase", "UPPERCASE", "Title Case"])
    if case is None:
        return
    new = df.copy()
    for c in cols:
        x = new[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        if case == "lowercase":
            x = x.str.lower()
        elif case == "UPPERCASE":
            x = x.str.upper()
        elif case == "Title Case":
            x = x.str.title()
        new[c] = x
    s.apply(new, f"clean_text {cols} case={case}")


def convert_type(s):
    cols = pick_columns(s.df, "Column to convert")
    if not cols:
        return
    t = choose("Convert to", ["number", "text", "date", "category"])
    if t is None:
        return
    new = s.df.copy()
    for c in cols:
        if t == "number":
            new[c] = pd.to_numeric(new[c], errors="coerce")
        elif t == "text":
            new[c] = new[c].astype("string")
        elif t == "date":
            new[c] = pd.to_datetime(new[c], errors="coerce")
        else:
            new[c] = new[c].astype("category")
    s.apply(new, f"convert {cols} to {t} (unparseable -> NaN)")


def outliers(s):
    df = s.df
    cols = pick_columns(df, "Numeric column(s) (Enter = all numeric)") or numeric_cols(df)
    method = choose("Detection", ["IQR (1.5 x IQR)", "Z-score (|z| > 3)"])
    if method is None:
        return
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    bounds = {}
    for c in cols:
        x = df[c]
        if method.startswith("IQR"):
            q1, q3 = x.quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        else:
            lo, hi = x.mean() - 3 * x.std(), x.mean() + 3 * x.std()
        bounds[c] = (lo, hi)
        mask[c] = (x < lo) | (x > hi)
    print("  Outliers per column:", mask.sum().to_dict())
    action = choose("What to do", ["only report (no change)", "remove rows containing outliers",
                                   "clip (winsorize) to bounds", "replace with NaN"])
    if action is None or action.startswith("only"):
        return
    new = df.copy()
    if action.startswith("remove"):
        new = new[~mask.any(axis=1)]
    elif action.startswith("clip"):
        for c in cols:
            new[c] = new[c].clip(*bounds[c])
    else:
        for c in cols:
            new.loc[mask[c], c] = np.nan
    s.apply(new, f"outliers {method} on {cols}: {action}")


def filter_rows(s):
    df = s.df
    col = ask("Column to filter on")
    if col not in df.columns:
        print("  Unknown column.")
        return
    op = choose("Condition", ["==", "!=", ">", ">=", "<", "<=", "between", "is in list",
                              "contains text", "is missing", "is not missing"])
    if op is None:
        return
    x = df[col]
    is_num = pd.api.types.is_numeric_dtype(x)
    conv = (lambda v: float(v)) if is_num else (lambda v: v.strip())
    try:
        if op == "is missing":
            m = x.isna()
        elif op == "is not missing":
            m = x.notna()
        elif op == "between":
            lo, hi = conv(ask("Lower bound (inclusive)")), conv(ask("Upper bound (inclusive)"))
            m = x.between(lo, hi)
        elif op == "is in list":
            vals = [conv(v) for v in ask("Values, comma-separated").split(",")]
            m = x.isin(vals)
        elif op == "contains text":
            m = x.astype("string").str.contains(ask("Text"), case=False, na=False)
        else:
            v = conv(ask("Value"))
            m = {"==": x == v, "!=": x != v, ">": x > v, ">=": x >= v, "<": x < v, "<=": x <= v}[op]
    except (ValueError, TypeError) as e:
        print(f"  Could not apply filter: {e}")
        return
    keep = ask("Keep rows that MATCH (k) or REMOVE rows that match (r)?", "k").lower()
    new = df[m] if keep.startswith("k") else df[~m]
    s.apply(new, f"filter {col} {op} (keep={'match' if keep.startswith('k') else 'non-match'})")


def columns_menu(s):
    action = choose("Action", ["keep only selected columns", "drop selected columns",
                               "rename a column", "add log1p column", "drop constant columns"])
    df = s.df
    if action is None:
        return
    if action.startswith("keep"):
        cols = pick_columns(df)
        if cols:
            s.apply(df[cols], f"keep columns {cols}")
    elif action.startswith("drop selected"):
        cols = pick_columns(df)
        if cols:
            s.apply(df.drop(columns=cols), f"drop columns {cols}")
    elif action.startswith("rename"):
        old = ask("Current name")
        if old in df.columns:
            new_name = ask("New name")
            s.apply(df.rename(columns={old: new_name}), f"rename {old} -> {new_name}")
    elif action.startswith("add log1p"):
        cols = [c for c in pick_columns(df) if c in numeric_cols(df)]
        new = df.copy()
        for c in cols:
            new[f"log_{c}"] = np.log1p(new[c].clip(lower=0))
        s.apply(new, f"log1p columns for {cols}")
    else:
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        s.apply(df.drop(columns=const), f"dropped constant columns {const}")


def save(s):
    path = ask("Output Excel file name", "cleaned_data.xlsx")
    if not path.lower().endswith(".xlsx"):
        path = path.rsplit(".", 1)[0] + ".xlsx" if "." in path.split("/")[-1] else path + ".xlsx"
    try:
        s.df.to_excel(path, index=False)  # needs openpyxl
        print(f"  Saved {len(s.df)} rows to {path}")
    except Exception as e:
        print(f"  Save failed: {e}")


def show_log(s):
    print("\nActions so far:" if s.log else "  No actions yet.")
    for i, n in enumerate(s.log, 1):
        print(f"  {i}. {n}")


# ---------- main ----------
def load(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


MENU = [
    ("Profile data (types, missing, stats)", profile),
    ("Preview rows", preview),
    ("Remove duplicates", drop_duplicates),
    ("Handle missing values", handle_missing),
    ("Clean text columns", clean_text),
    ("Convert column types", convert_type),
    ("Detect / handle outliers", outliers),
    ("Filter rows", filter_rows),
    ("Columns (keep/drop/rename/log/constant)", columns_menu),
    ("Show action log", show_log),
    ("Undo last step", lambda s: s.undo()),
    ("Save cleaned data", save),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ask("Path to CSV/Excel file")
    try:
        s = Session(load(path))
    except Exception as e:
        print(f"Could not load file: {e}")
        return
    print(f"Loaded {s.df.shape[0]} rows x {s.df.shape[1]} cols")
    while True:
        print("\n=== MENU ===")
        for i, (name, _) in enumerate(MENU, 1):
            print(f"  {i}) {name}")
        print("  0) Quit")
        c = ask("Choose")
        if c == "0":
            if s.log and ask("Save before quitting? (y/n)", "y").lower().startswith("y"):
                save(s)
            break
        if c and c.isdigit() and 1 <= int(c) <= len(MENU):
            try:
                MENU[int(c) - 1][1](s)
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExited.")