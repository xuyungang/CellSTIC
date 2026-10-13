import os

import torch


class EnvironmentUtils:
    """Device and R environment helpers for baseline runners."""

    @staticmethod
    def get_device(device_id: int) -> torch.device:
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
            else:
                print(f"Warning: CUDA device {device_id} not available. Using cuda:0 instead.")
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            if device_id is not None:
                print(f"Warning: CUDA not available. Using CPU instead of requested device {device_id}.")
        print(f"Using device: {device}")
        return device

    @staticmethod
    def set_r_environment(r_home: str) -> None:
        r_home_actual = os.path.join(r_home, "lib", "R") if os.path.exists(os.path.join(r_home, "lib", "R")) else r_home
        os.environ["R_HOME"] = r_home_actual

        r_lib_path = os.path.join(r_home_actual, "lib")
        r_libs = os.path.join(r_home_actual, "library")

        if os.path.exists(r_lib_path):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{r_lib_path}:{current_ld_path}" if current_ld_path else r_lib_path

        if os.path.exists(r_libs):
            os.environ["R_LIBS"] = r_libs

        renviron = os.path.join(r_home_actual, "etc", "Renviron")
        if os.path.exists(renviron):
            os.environ["R_ENVIRON"] = renviron
            os.environ["R_ENVIRON_USER"] = renviron

    @staticmethod
    def get_r_environment_for_subprocess(r_home: str = None) -> dict:
        if r_home:
            EnvironmentUtils.set_r_environment(r_home)
            os.environ["R_HOME"] = r_home
            r_bin = os.path.join(r_home, "bin")
            if "PATH" in os.environ:
                os.environ["PATH"] = f"{r_bin}:{os.environ['PATH']}"
            else:
                os.environ["PATH"] = r_bin
        return os.environ.copy()
