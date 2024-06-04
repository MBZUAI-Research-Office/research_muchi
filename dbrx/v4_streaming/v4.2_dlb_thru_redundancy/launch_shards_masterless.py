# Examples:
#  launch all shards:
#    python launch_shards_masterless.py --model-path ~/dbrx-base/distributable/batch2/
#
#  terminate all shards:
#    python launch_shards_masterless.py --model-path ~/dbrx-base/distributable/batch2/ --terminate
import subprocess
import time
from types import SimpleNamespace
import argparse
from pathlib import Path
import json

"""terminal color"""
TC = SimpleNamespace(
    **{
        "YELLOW": "\033[33m",
        "GREEN": "\033[92m",
        "RED": "\033[91m",
        "BLUE": "\033[34m",
        "RESET": "\033[0m",
    }
)


class Cmd:
    def __new__(
        self, cmd: str, cwd="./", timeout_duration=None, suppress=True
    ) -> tuple[int, str, str]:
        self.cmd = cmd
        self.cwd = cwd
        self.returncode = 0
        self.has_err = True

        if not suppress:
            print(f"{self.cmd}", end="", flush=True)
        cwd_not_cur = f" in {self.cwd}" if self.cwd != "./" else ""

        """ process setup """
        process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="bash",
            cwd=self.cwd,
        )

        """ timeout """
        # https://stackoverflow.com/a/13821695
        import signal

        class TimeoutError(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutError()

        # set the timeout handler
        if timeout_duration is not None:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_duration)

        """ execution """
        out = bytearray()
        err = bytearray()
        timeStarted = time.time()
        try:
            _out, _err = process.communicate()
            out = _out if _out is not None else out
            err = _err if _err is not None else err
            self.returncode = process.returncode
            if process.returncode != 0:
                raise RuntimeError(
                    f"returncode is not 0 but {process.returncode}. "
                    + str(out + err, encoding="utf8")
                )
        except RuntimeError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except TimeoutError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        finally:  # reset timeout handler
            signal.alarm(0)

        timeDelta = time.time() - timeStarted
        if not suppress:
            print(f"{cwd_not_cur} {TC.GREEN}[passed]{TC.RESET} ({timeDelta:.3f}s)")
        self.has_err = False
        return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", required=True, type=str, help="Path to local model directory"
    )
    parser.add_argument("--terminate", action="store_true")
    args = parser.parse_args()
    try:
        with open(Path(args.model_path) / "v4.2_broker_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise

    for url in config["shard_urls"]:
        pure_url, port = url.split(":")
        print(f"Shard: {pure_url} {port} ", end="")
        Cmd(
            f"""scp -i ~/.ssh/id_llamacpp ./run_shard_masterless.py xiangruike@{pure_url}:/users/xiangruike"""
        )
        if args.terminate:
            rc, out, err = Cmd(
                f"""ssh -i ~/.ssh/id_llamacpp xiangruike@{pure_url} 'export PATH="$PATH:/opt/homebrew/bin/" """
                + f"""&& python3 /Users/xiangruike/run_shard_masterless.py --port {port} --terminate'"""
            )
            if rc != 0:
                print(err.strip())
            else:
                print("[terminated successfully]")
        else:
            rc, out, err = Cmd(
                f"""ssh -i ~/.ssh/id_llamacpp xiangruike@{pure_url} 'export PATH="$PATH:/opt/homebrew/bin/" """
                + f"""&& python3 /Users/xiangruike/run_shard_masterless.py --port {port}'"""
            )
            if rc != 0:
                print(err.strip())
            else:
                print("[launched successfully]")


if __name__ == "__main__":
    main()
