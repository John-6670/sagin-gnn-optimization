import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from models.gnn.dataset import get_gnn_datasets


NUM_PROCESSES = 10
SAMPLES_PER_PROCESS = 10
BASE_SEED = 42

OUTPUT_DIR = "precomputed_dataset"


def generate_samples(worker_id):
    """
    Generate an independent batch of samples in a separate process.
    """
    seed = BASE_SEED + worker_id

    print(
        f"[Worker {worker_id}] Starting "
        f"{SAMPLES_PER_PROCESS} samples with seed={seed}"
    )

    train_ds, val_ds, test_ds = get_gnn_datasets(
        num_samples=SAMPLES_PER_PROCESS,
        seed=seed
    )

    print(f"[Worker {worker_id}] Finished")

    return worker_id, train_ds, val_ds, test_ds


if __name__ == "__main__":

    # Remove previous dataset
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Storage for results from all workers
    all_train = []
    all_val = []
    all_test = []

    print(
        f"Starting {NUM_PROCESSES} processes, "
        f"{SAMPLES_PER_PROCESS} samples each..."
    )

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:

        futures = [
            executor.submit(generate_samples, worker_id)
            for worker_id in range(NUM_PROCESSES)
        ]

        results = []

        # Wait for every process
        for future in as_completed(futures):
            worker_id, train_ds, val_ds, test_ds = future.result()

            results.append(
                (worker_id, train_ds, val_ds, test_ds)
            )

    # Keep the workers in deterministic order
    results.sort(key=lambda x: x[0])

    # Combine everything
    for worker_id, train_ds, val_ds, test_ds in results:
        all_train.extend(train_ds)
        all_val.extend(val_ds)
        all_test.extend(test_ds)

    print("All workers finished.")
    print(f"Total train samples: {len(all_train)}")
    print(f"Total validation samples: {len(all_val)}")
    print(f"Total test samples: {len(all_test)}")

    # Save exactly like the original script
    for i, data in enumerate(all_train):
        torch.save(
            data,
            f"{OUTPUT_DIR}/sample_train_{i}.pt"
        )

    for i, data in enumerate(all_val):
        torch.save(
            data,
            f"{OUTPUT_DIR}/sample_val_{i}.pt"
        )

    for i, data in enumerate(all_test):
        torch.save(
            data,
            f"{OUTPUT_DIR}/sample_test_{i}.pt"
        )

    print("Precomputation done: 6000 samples created.")