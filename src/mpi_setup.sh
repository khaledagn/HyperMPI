#!/bin/bash

# the banner
display_banner() {
  clear
  echo "====================================================================="
  echo "       __  __ _____ _____  _______      _       _                 _      "
  echo "      |  \/  |_   _|  __ \|__   __|    | |     | |               | |     "
  echo "      | \  / | | | | |  | |  | |  __ _| |_ ___| |__  _   _  __ _| | __ _ "
  echo "      | |\/| | | | | |  | |  | | / _\` | __/ __| '_ \| | | |/ _\` | |/ _\` |"
  echo "      | |  | |_| |_| |__| |  | || (_| | || (__| | | | |_| | (_| | | (_| |"
  echo "      |_|  |_|_____|_____/   |_| \__,_|\__\___|_| |_|\__,_|\__, |_|\__,_|"
  echo "                                                         __/ |          "
  echo "                                                        |___/           "
  echo "====================================================================="
  echo " Distributed Training Setup Script"
  echo " Automates the setup of SSH keys, MPI, and Python environment across multiple hosts."
  echo " Copyright © 2024 Khaled AGN and Yasmine Aoui"
  echo "====================================================================="
}

# display the banner
display_banner

# validate ip address
validate_ip() {
  local ip=$1
  local valid_ip_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
  if [[ $ip =~ $valid_ip_regex ]]; then
    IFS='.' read -r -a octets <<< "$ip"
    for octet in "${octets[@]}"; do
      if ((octet > 255)); then
        return 1
      fi
    done
    return 0
  else
    return 1
  fi
}

# check if the script is running on the main server
read -p "Is this the MAIN HOST (yes/no)? " main_host
if [[ "$main_host" != "yes" ]]; then
  echo "This script must be run on the MAIN HOST"
  exit 1
fi

declare -A root_passwords

# create the host file for MPI
create_host_file() {
  display_banner
  echo "Creating MPI host file..."

  # get the absolute path of the project folder
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
  CONFIG_DIR="$PROJECT_DIR/config"

  # ensure the config directory exists
  if [ ! -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR"
    echo "Created missing config directory."
  fi

  # remove existing hostfile if it exists
  HOSTFILE="$CONFIG_DIR/hostfile"
  if [ -f "$HOSTFILE" ]; then
    echo "Removing existing hostfile..."
    rm -f "$HOSTFILE"
  fi

  read -p "Enter the number of hosts (minimum 2): " num_hosts
  if ((num_hosts < 2)); then
    echo "A minimum of 2 machines is required to run this script."
    exit 1
  fi

  for ((i=1; i<=num_hosts; i++)); do
    while true; do
      read -p "Enter IP address for host $i: " ip_address
      if validate_ip "$ip_address"; then
        break
      else
        echo "Invalid IP address format. Please enter a valid IP address."
      fi
    done
    read -sp "Enter root password for host $i: " root_password
    echo
    root_passwords[$ip_address]=$root_password
    while true; do
      read -p "Enter number of slots for host $i: " slots
      if [[ $slots =~ ^[0-9]+$ ]]; then
        break
      else
        echo "Invalid slots number. Please enter a valid number."
      fi
    done
    echo "$ip_address slots=$slots" >> "$HOSTFILE"
  done

  # validate if the hostfile was created and not empty
  if [ -s "$HOSTFILE" ]; then
    echo "MPI host file created successfully at $HOSTFILE"
  else
    echo "Error: MPI host file was not created or is empty."
    exit 1
  fi
}





# setup SSH keys and copy them between servers
setup_ssh_keys() {
  display_banner
  echo "Setting up SSH keys..."

  # read host IP addresses
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
  CONFIG_DIR="$PROJECT_DIR/config"
  HOSTFILE="$CONFIG_DIR/hostfile"
  mapfile -t hosts < <(awk '{print $1}' $HOSTFILE)


  # remove existing SSH keys and generate new ones on all hosts
  for ip_address in "${hosts[@]}"; do
    echo "Removing existing SSH keys on $ip_address..."
    sshpass -p "${root_passwords[$ip_address]}" ssh -o StrictHostKeyChecking=no root@$ip_address "rm -f ~/.ssh/id_rsa ~/.ssh/id_rsa.pub; rm -f ~/.ssh/authorized_keys"
    echo "Generating new SSH key on $ip_address..."
    sshpass -p "${root_passwords[$ip_address]}" ssh -o StrictHostKeyChecking=no root@$ip_address "ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa"
  done

  # copy SSH keys to all hosts and remove old authorized keys
  for src_ip in "${hosts[@]}"; do
    echo "Copying SSH key to $src_ip..."
    ssh-keygen -R $src_ip
    sshpass -p "${root_passwords[$src_ip]}" ssh-copy-id -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa.pub root@$src_ip
    for dest_ip in "${hosts[@]}"; do
      if [ "$src_ip" != "$dest_ip" ]; then
        echo "Copying SSH key from $src_ip to $dest_ip..."
        ssh-keygen -R $dest_ip
        sshpass -p "${root_passwords[$src_ip]}" ssh -o StrictHostKeyChecking=no root@$src_ip "sshpass -p '${root_passwords[$dest_ip]}' ssh-copy-id -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa.pub root@$dest_ip"
      fi
    done
  done

  echo "SSH keys setup completed."
}



# install necessary system packages
install_system_packages() {
  display_banner
  echo "Installing necessary system packages on $1..."
  sshpass -p "${root_passwords[$1]}" ssh -o StrictHostKeyChecking=no root@$1 "apt-get update && apt-get install -y python3 pip wget build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev sshpass libffi-dev unzip nano mpich lam-runtime slurm-wlm-torque openmpi-bin openmpi-common openmpi-doc libopenmpi-dev"
  echo "System packages installation on $1 completed."
}

# install necessary pip packages
install_pip_packages() {
  display_banner
  echo "Installing pip packages on $1..."
  sshpass -p "${root_passwords[$1]}" ssh -o StrictHostKeyChecking=no root@$1 "sudo python3 -m pip install mpi4py tensorflow numpy matplotlib pandas scipy scikit-learn jupyter seaborn pillow h5py " || { echo "Failed to install pip packages on $1"; exit 1; }
  echo "Pip packages installation on $1 completed."
}

# download and set up the GitHub project in the proper directories
setup_github_project() {
  display_banner
  echo "Downloading and setting up GitHub project on $1..."
  sshpass -p "${root_passwords[$1]}" ssh -o StrictHostKeyChecking=no root@$1 "git clone https://github.com/khaledagn/HyperMPI.git"
  echo "GitHub project setup completed on $1."
}

# execute the functions on all hosts
create_host_file
setup_ssh_keys

mapfile -t hosts < <(awk '{print $1}' ../config/hostfile)

for host in "${hosts[@]}"; do
  display_banner
  echo "Starting installation on $host..."
  echo "----------------------------------------"
  echo "Installing necessary system packages on $host"
  install_system_packages $host
  echo "Installing necessary pip packages on $host"
  install_pip_packages $host
  echo "Downloading and setting up GitHub project on $host"
  setup_github_project $host
  echo "----------------------------------------"
  echo "Completed installation on $host"
done

echo "All tasks completed."
