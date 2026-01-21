import subprocess, re, sys, json, pathlib

IN_YAML = "environments/cu128.yml"
OUT_YAML = "environments/cu128.updated_to_current.yml"

# Get current env package versions/builds
pkgs = subprocess.check_output(["conda", "list", "--json"], text=True)
pkgs = json.loads(pkgs)
cur = {p["name"].lower(): (p["version"], p["build_string"]) for p in pkgs}

yaml_text = pathlib.Path(IN_YAML).read_text()

out_lines = []
in_deps = False

# dependency line matcher for lines like: "  - pkg=1.2.3=build"
dep_pat = re.compile(r'^(\s*-\s*)([A-Za-z0-9_.+-]+)(?:=([^=]+)=([^\s#]+))?(.*)$')

for line in yaml_text.splitlines(True):
    if line.strip() == "dependencies:":
        in_deps = True
        out_lines.append(line)
        continue

    if in_deps:
        m = dep_pat.match(line)
        if m:
            prefix, name, oldver, oldbuild, suffix = m.groups()
            key = name.lower()

            # If package exists in current env, rewrite pin to current version/build
            if key in cur:
                ver, build = cur[key]
                line = f"{prefix}{name}={ver}={build}{suffix}\n"
            # else: keep line as-is (it may be missing and should be installed)

    out_lines.append(line)

pathlib.Path(OUT_YAML).write_text("".join(out_lines))
print(f"Wrote: {OUT_YAML}")