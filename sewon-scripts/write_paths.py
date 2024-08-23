


with open("output.txt", "w") as f:
    for shard_idx in range(6):
        for file_idx in range(50):
            f.write("    - /weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/SD-B34v0/shard{}/part-00-{}.npy\n".format(shard_idx, str(file_idx).zfill(5)))


