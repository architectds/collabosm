"""Runs ON the Colab VM: did bootstrap finish, and did it fail?"""
import os

log = "/content/bootstrap.log"
if not os.path.exists(log):
    print("bootstrap_not_started")
    raise SystemExit(0)
txt = open(log, errors="replace").read()
if "BOOTSTRAP_FAILED" in txt:
    print("bootstrap_failed")
    print(txt[-4000:])
elif "BOOTSTRAP_OK" in txt:
    print("bootstrap_ok")
else:
    print("bootstrap_running")
    print(txt[-400:])