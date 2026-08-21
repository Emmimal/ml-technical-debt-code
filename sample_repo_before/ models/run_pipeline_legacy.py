import subprocess
import shutil
import os

def run_all():
    subprocess.run(["python", "feature_gen.py"], check=True)
    subprocess.run(["python", "train.py"], check=True)
    subprocess.run(["python", "score_batch.py"], check=True)
    shutil.copy("data/scores_output.csv", "archive/scores_backup_v2_old.csv")
    os.system("echo pipeline done >> logs/run.log")

if __name__ == "__main__":
    run_all()
