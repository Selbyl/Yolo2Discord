from ultralytics import YOLO
import datetime
import socket
import requests
import subprocess
import time
import csv
import os

KMP_Duplicate_LIB_OK=True

# Discord webhook settings
WEBHOOK_URL = 'Insert Webhook URL'  # replace with your webhook URL

def send_discord_message(content):
    """
    Send a message to the Discord server via the specified webhook.
    """
    # Split content into chunks of 1900 characters (to leave some buffer)
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    
    for chunk in chunks:
        data = {"content": chunk}
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code != 204:
            print(f"Failed to send chunk to Discord. Status code: {response.status_code}")

# Load a model
model = YOLO('Yolov8n.pt')  # build a new model or use yaml

# Get the current date and hostname
current_date_time = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
hostname = socket.gethostname()

# Send a message to Discord indicating that training has started
start_time = time.time()
send_discord_message(f"Training has started on {hostname} at {current_date_time}.")

# Train the model
total_epochs = 999999
training_name = f"{hostname}_{current_date_time}"
results_path = f"./runs/detect/{training_name}/results.csv"

# Assuming model.train() trains for all epochs at once
model.train(data='./project/dataset.yaml', epochs=999999, imgsz=3400, device=[0,1], augment=True, nbs=256, batch=6, name=training_name, cache='ram', workers=32, save_period=10, patience=1000)

# After training is complete, send final metrics to Discord
elapsed_time = time.time() - start_time
send_discord_message(f"Training completed in {elapsed_time:.2f} seconds. Final results can be found at {results_path}.")

# Fetch and send results for every 5th epoch
if os.path.exists(results_path):
    with open(results_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        headers = csv_reader.fieldnames
        for epoch, row in enumerate(csv_reader):
            if (epoch + 1) % 5 == 0:
                # Get NVIDIA GPU stats
                nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")

                # Compile the stats message
                message = f"Epoch {epoch + 1}/{total_epochs} stats:\n"
                for header in headers:
                    message += f"{header}: {row[header]}\n"
                message += f"\nNVIDIA-SMI output:\n{nvidia_smi_output}"

                # Send the stats to Discord
                send_discord_message(message)
else:
    send_discord_message(f"Could not find results at {results_path}. Training may not have completed successfully.")
