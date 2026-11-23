def parse_value(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return text.strip("'")


def load_tir(path):
    params = {}
    section = None

    with open(path) as f:
        for line in f:
            line = line.split("$")[0].strip()

            if line == "" or line.startswith("!"):
                continue

            if line.startswith("["):
                section = line.strip("[]")
                params[section] = {}
                continue

            key, value = line.split("=")
            params[section][key.strip()] = parse_value(value)

    return params
