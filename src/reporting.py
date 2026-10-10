"""Save experiment results as an Excel file, a text file and a chart."""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_results(folder, title, tables, settings, notes=None):
    """Write folder/results.xlsx (one sheet per table + a 'settings' sheet) and folder/results.txt.

    tables   : {"sheet name": DataFrame}
    settings : {"setting": value}   (seed, epochs, image size, split id, ...)
    notes    : list of text lines added at the end of the text file
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- Excel
    with pd.ExcelWriter(folder / "results.xlsx", engine="openpyxl") as writer:
        settings_table = pd.DataFrame({"value": {k: str(v) for k, v in settings.items()}})
        settings_table.to_excel(writer, sheet_name="settings")
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet[:31])
        for sheet in writer.sheets.values():                                   # widen the columns so the file is readable
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(width + 2, 60)

    # ---------------- text
    lines = [title, "=" * len(title), f"created: {stamp}", "", "SETTINGS"]
    lines += [f"  {k}: {v}" for k, v in settings.items()]
    for sheet, table in tables.items():
        lines += ["", sheet.upper(), table.to_string(float_format=lambda v: f"{v:.3f}", max_colwidth=40)]
    if notes:
        lines += ["", "NOTES"] + [f"  - {n}" for n in notes]
    (folder / "results.txt").write_text("\n".join(lines) + "\n")
    return folder / "results.xlsx", folder / "results.txt"


def save_bar_chart(table, path, title, ylabel):
    """Grouped bar chart of a table (rows on the x axis, one bar per column)."""
    ax = table.plot.bar(figsize=(9, 4), title=title)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_scatter(x, y, labels, path, title, xlabel, ylabel):
    """Scatter plot with a text label next to every point (used for accuracy-versus-speed in E2)."""
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    for xi, yi, text in zip(x, y, labels):
        plt.annotate(text, (xi, yi), textcoords="offset points", xytext=(6, 6))
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
