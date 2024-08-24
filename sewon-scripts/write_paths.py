


with open("output.txt", "w") as f:
    for repeat_idx in range(10):
        f.write("    # {}\n".format(repeat_idx))
        for shard_idx in range(5):
            for file_idx in range(10):
                f.write("    - /weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/SD-B34v0/shard{}/part-{}-00000.npy\n".format(shard_idx, str(file_idx).zfill(2)))
        f.write("\n")

