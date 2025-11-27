import pathlib

data = {}
with pathlib.Path("tree_output_css.txt").open() as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line not in {
            "{",
            "}",
            "[",
            "]",
            ";",
            "(",
            ")",
            ".",
            "=",
            "-=",
            "===",
            "!",
            "'",
            ">=",
            "&&",
            "<=",
            ",",
            ":",
            "!=",
            "=>",
            "!==",
            "++",
            "+=",
            "?",
            "==",
            ">",
            "%",
            "<",
            "${",
            "-",
            "*",
            '"',
            "`",
            "+",
            "/",
            "||",
        }:
            if line not in data:
                data[line] = 1
            else:
                data[line] += 1
print(data)
