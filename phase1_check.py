"""Run from the project directory with the Python environment used for training."""
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

report = {"python": sys.version.split()[0], "platform": platform.platform()}
checks = {}


def check(name, run):
    try:
        checks[name] = {"status": "PASS", "detail": run()}
    except Exception as error:
        checks[name] = {"status": "FAIL", "detail": str(error)}
    print(name, checks[name]["status"], checks[name]["detail"])


def c_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "probe.c"
        binary = root / "probe"
        source.write_text("""
#include <stdint.h>
#include <stdio.h>
int main(void) {
    const int8_t signal[] = {4, 9, -3};
    const int8_t weights[] = {-1, 0, 1};
    int32_t sum = 0;
    for (int i = 0; i < 3; ++i) {
        if (weights[i] == 1) sum += signal[i];
        if (weights[i] == -1) sum -= signal[i];
    }
    printf("%d", (int)sum);
    return 0;
}
""")
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        str(source), "-o", str(binary)], check=True)
        value = subprocess.check_output([str(binary)], text=True)
        if value != "-7":
            raise ValueError(f"Expected -7, got {value}")
        return subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]


def torch_test():
    import torch
    x = torch.tensor([[[1., 2., 4.]]], device="cpu")
    kernel = torch.tensor([[[-1., 1.]]], requires_grad=True, device="cpu")
    y = torch.nn.functional.conv1d(x, kernel)
    torch.testing.assert_close(y, torch.tensor([[[1., 2.]]]))
    y.sum().backward()
    torch.testing.assert_close(kernel.grad, torch.tensor([[[3., 6.]]]))
    return str(torch.__version__)


if __name__ == "__main__":
    check("desktop_c", c_test)
    check("pytorch_cpu_forward_backward", torch_test)
    report["checks"] = checks
    passed = all(item["status"] == "PASS" for item in checks.values())
    report["status"] = "PASS" if passed else "FAIL"
    output = Path("results/phase1_environment.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Phase 1", report["status"], "|", output)
    sys.exit(0 if passed else 1)
