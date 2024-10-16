import ladder_peteish as ladder

tps = {"190M": 74000, "370M": 43000, "600M": 34000, "760M": 32000, "1B": 20000}
nodes = {"190M": 4, "370M": 8, "600M": 8, "760M": 8, "1B": 16}


def how_long(length, size):
    return (ladder.parse_length(length, ladder.MODEL_PARAMS[size]) / (tps[size] * 8 * nodes[size])) / 3600


if __name__ == "__main__":
    total_hours = 0
    for key in tps:
        for length in ["1xC", "2xC", "5xC", "10xC"]:
            total_hours += how_long(length, key)

    print("Total GPU hours:", round(total_hours))
