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



def write_tir(template_path, out_path, updates):
    section = None
    lines_out = []

    with open(template_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                section = stripped.split("$")[0].strip().strip("[]")
            elif "=" in line and not stripped.startswith(("$", "!")):
                key = line.split("=")[0].strip()
                if section in updates and key in updates[section]:
                    comment = ""
                    if "$" in line:
                        comment = "  $" + line.split("$", 1)[1].rstrip("\n")
                    line = f"{key:<25}= {updates[section][key]:.8g}{comment}\n"
            lines_out.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines_out)