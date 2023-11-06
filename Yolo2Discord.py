from ultralytics import YOLO
import datetime
import socket
import requests
import subprocess
import time
import csv
import os
import threading

KMP_Duplicate_LIB_OK = True

# Discord webhook settings
WEBHOOK_URL = ''  # replace with your webhook URL

def send_discord_message(content):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    for chunk in chunks:
        data = {"content": chunk}
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code != 204:
            print(f"Failed to send chunk to Discord. Status code: {response.status_code}")

def format_time(seconds_elapsed):
    days, remainder = divmod(seconds_elapsed, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def get_newest_folder_path(base_path):
    folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return max(folders, key=os.path.getctime)

def metrics_sender():
    while not training_complete:
        time.sleep(300)  # wait for 15 minutes

        results_folder_path = get_newest_folder_path('./runs/detect/')
        results_file_path = os.path.join(results_folder_path, 'results.csv')

        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                headers = csv_reader.fieldnames
                last_row = list(csv_reader)[-1]  # get the last row of the csv file
                
                # Get NVIDIA GPU stats
                nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
                
                message = "Results:\n"
                for header in headers:
                    message += f"{header}: {last_row[header]}\n"
                
                message += f"\nNVIDIA-SMI output:\n{nvidia_smi_output}"
                
                # This will automatically split the message into 1900 character chunks and send them
                send_discord_message(message)
        else:
            send_discord_message(f"Could not find results at {results_file_path}. Training may still be ongoing.")

# Load a model
model = YOLO('.yamlv8n.pt') #input model name or directory of previous model

# Get the current date and hostname
current_date_time = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
hostname = socket.gethostname()

# Send a message to Discord indicating that training has started
start_time = time.time()
send_discord_message(f"Training has started on {hostname} at {current_date_time}.")

# Train the model settings
total_epochs = 999999
training_name = f"{hostname}_{current_date_time}"
results_path = f"./runs/detect/{training_name}/results.csv"

# Flag to indicate the completion of training. This will be checked by our background thread.
training_complete = False

# Start the metrics_sender function in a separate thread
sender_thread = threading.Thread(target=metrics_sender)
sender_thread.start()

# Train the model
model.train(data='./project/dataset.yaml', epochs=999999, imgsz=3400, device=[0,1], augment=True, nbs=256, batch=6, name=training_name, cache='ram', workers=32, save_period=10, patience=1000)

# Set the training_complete flag to True once training is finished
training_complete = True
sender_thread.join()  # Ensure that the metrics_sender thread has finished

# After training is complete, send final metrics to Discord
elapsed_time = time.time() - start_time
elapsed_time_str = format_time(elapsed_time)
send_discord_message(f"Training completed in {elapsed_time_str}. Final results can be found at {results_path}.")
