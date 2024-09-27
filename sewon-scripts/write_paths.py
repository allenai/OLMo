

old_path = "/weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/SD-B34v0.1/"
new_path = "/weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/books3/"

with open("sewon-configs/peteish7-anneal-B3x50-weka.yaml", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    lines[i] = line.replace(old_path, new_path)

with open("sewon-configs/peteish7-anneal-B3x50-weka.yaml", "w") as f:
    for line in lines:
        f.write(line)



