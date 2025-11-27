import re

data = {}
with open("tree_output_css.txt" ,"r") as f:
  lines = f.readlines()
  for line in lines:
    line = line.strip()
    if  line not in {"{","}", "[","]",";", "(",")", ".", "=", "-=","===","!", "'", '>=', '&&', '<=',",",":","!=", "=>", "!==", "++", "+=", "?", "==",">", "%", "<", '${', '-',"*", '"',"`", "+", "/", "||"}:
      if line not in data:
        data[line] = 1
      else:
        data[line] += 1
print(data)

