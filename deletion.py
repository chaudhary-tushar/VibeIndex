import json
from pathlib import Path

tree_path = Path("data/ct_symbols/ct_symbols.json")
# data = {}
# with Path(tree_path).open() as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         if line not in {
#             "{",
#             "}",
#             "[",
#             "]",
#             ";",
#             "(",
#             ")",
#             ".",
#             "=",
#             "-=",
#             "===",
#             "!",
#             "'",
#             ">=",
#             "&&",
#             "<=",
#             ",",
#             ":",
#             "!=",
#             "=>",
#             "!==",
#             "++",
#             "+=",
#             "?",
#             "==",
#             ">",
#             "%",
#             "<",
#             "${",
#             "-",
#             "*",
#             '"',
#             "`",
#             "+",
#             "/",
#             "||",
#         }:
#             if line not in data:
#                 data[line] = 1
#             else:
#                 data[line] += 1
# print(data)

with Path(tree_path).open("r") as f:
    symbols = json.load(f)

print(type(symbols))
print(type(symbols[0]))
print(type(symbols[0][0]))
data = {}
for sym in symbols:
    for s in sym:
        for key in s.keys():
            if key in data:
                data[key] += 1
            else:
                data[key] = 1
print(data)
