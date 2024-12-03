
MODEL_FLOPS = {
    "190M": 1903391232,
    "370M": 3443922944,
    "600M": 5180751744,
    "760M": 6373843968,
    "1B": 10109071360,
    "3B": 22970355200,
    "7B": 49412071424,
    "13B": 91335915520,
}

MODEL_PARAMS = {
    "190M": 190354176,
    "370M": 371262464,
    "600M": 597382464,
    "760M": 758220288,
    "1B": 1279395840,
    "3B": 3169537280,
    "7B": 6887575552,
    "13B": 13202396160,
}

def compute_flops():

	ladder_flops = 0

	for N in ["190M", "370M", "760M", "1B"]:
		for mult in [1, 2, 5, 10]:
			flops = MODEL_FLOPS[N] * (MODEL_PARAMS[N] * 20 * mult)
			ladder_flops += flops

	target_flops_7B = MODEL_FLOPS["7B"] * 4 * 10**12

	print(f"Compute needed to predict 7B: {round(ladder_flops * 100 / target_flops_7B, 3)} %")

	target_flops_13B = MODEL_FLOPS["13B"] * 5 * 10**12

	print(f"Compute needed to predict 13B: {round(ladder_flops * 100 / target_flops_13B, 3)} %")

	print(f"Compute needed to predict both target models: {round(ladder_flops * 100 / (target_flops_7B + target_flops_13B), 3)} %")

if __name__ == "__main__":
	compute_flops()

	# Output:
	# Compute needed to predict 7B: 3.535 %
	# Compute needed to predict 13B: 1.53 %
	# Compute needed to predict both target models: 1.068 %
