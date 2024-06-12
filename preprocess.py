import os
from scapy.all import rdpcap
from scapy.layers.inet import IP
import pickle
import ipaddress
from concurrent.futures import ProcessPoolExecutor
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def is_private_ip(ip):
    return ipaddress.ip_address(ip).is_private

def pcap_to_feature_vector(pcap_file):
    logger.info(f"Processing file: {pcap_file}")
    try:
        packets = rdpcap(pcap_file)
        features = []
        arr_len = 0
        for packet in packets:
            if IP in packet:
                src_ip = packet[IP].src
                direction = 1 if is_private_ip(src_ip) else -1
                features.append(direction)
                arr_len += 1
            if arr_len == 5000:
                break
        if arr_len < 5000:
            features += [0] * (5000 - arr_len)
        return features
    except Exception as e:
        logger.error(f"Error processing file {pcap_file}: {e}")
        return None  # Return None to indicate that processing failed


def process_directory(input_subdir, output_dir):
    output_subdir = os.path.join(
        output_dir, os.path.basename(input_subdir)
    )  # Fix to use basename
    all_pcap_paths = []

    for pcap_file in os.listdir(input_subdir):
        pcap_path = os.path.join(input_subdir, pcap_file)
        all_pcap_paths.append(pcap_path)

    feature_vectors = []

    for pcap_path in all_pcap_paths:
        feature_vector = pcap_to_feature_vector(pcap_path)
        if feature_vector is not None:
            feature_vectors.append(feature_vector)
        else:
            logger.warning(f"Skipping file due to processing error: {pcap_path}")

    logger.info(f"Processed {len(feature_vectors)} files in {input_subdir}")

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    output_file = os.path.join(output_subdir, "features.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(feature_vectors, f)

    logger.info(f"Saved processed data to {output_file}")


def process_directories_parallel(dataset_path, output_directory, max_workers):
    logger.info("Starting to process pcap files...")

    subdirs = sorted(
        [
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_directory, subdir, output_directory)
            for subdir in subdirs
        ]
        for future in futures:
            future.result()  # Ensure all futures are completed

    logger.info("Finished processing pcap files.")


# Command line arguments
if len(sys.argv) != 4:
    print(
        "Usage: python preprocess.py <input_directory> <output_directory> <max_threads>"
    )
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
max_threads = int(sys.argv[3])

process_directories_parallel(input_directory, output_directory, max_threads)

logger.info("All done!")
