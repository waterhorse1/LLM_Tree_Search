import requests

def get_worker_address(model_name: str, controller_addr: str):
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    return worker_addr