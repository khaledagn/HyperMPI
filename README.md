## HyperMPI Distributed Training System

HyperMPI is an automated system designed to set up distributed machine learning training environments across multiple hosts using MPI (Message Passing Interface). This tool configures SSH, installs necessary system packages, and sets up a Python environment for scalable training tasks across networks of machines.

## Features

🚀 Automated Multi-Host Setup: Easily configure SSH keys and MPI for distributed computing.

🔧 OpenMPI Installation: Handles OpenMPI installation across multiple hosts.

🐍 Python Environment: Automated Python environment setup with mpi4py and tensorflow.

📦 Customizable Host Files: Flexible multi-host configurations for training scalability.

📂 Project Auto-Download: Downloads and configures project files from GitHub.


## Installation

Follow the simple steps to install and run HyperMPI:

Clone the repository:

```
git clone https://github.com/khaledagn/HyperMPI.git && cd HyperMPI
```

Run the setup script:

```
sudo apt update && sudo apt install sshpass -y && chmod +x src/mpi_setup.sh && ./src/mpi_setup.sh
```

Follow the instructions to set up your multi-host distributed training environment.


## How to Use

* Create Hostfile
The setup script will help you create a hostfile, specifying IP addresses and slots for each host.

* Set Up SSH Keys
Easily configure SSH keys to enable secure, password-less communication between hosts.

* Install Required Packages
The script installs system packages such as OpenMPI, Python 3.12, and mpi4py.

* Run Distributed Training
Once the setup is complete, run your distributed training with the following command:

```
mpiexec --allow-run-as-root --oversubscribe -n 6 --hostfile /path/HyperMPI/config/hostfile  python3 /path/HyperMPI/training/ParallelTrain.py
```

# Contributing

Feel free to open issues or pull requests. All contributions are welcome! Whether it's fixing bugs or adding new features, we appreciate your effort to improve HyperMPI.

## License
HyperMPI is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions, reach out to the developers:

Khaled AGN: contact@khaledagn.com

Yasmine Aoui: yasmine.aouim@gmail.com

© 2024 Khaled AGN & Yasmine Aoui.

