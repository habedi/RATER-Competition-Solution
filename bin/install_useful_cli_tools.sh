#!/bin/bash

# List of the packages to be installed and their descriptions
packages=(
  "ccze:A robust log colorizer with multiple color schemes"
  "htop:Interactive process viewer"
  "nano:Easy-to-use text editor"
  "wget:A free utility for non-interactive download of files from the web"
  "curl:A tool to transfer data from or to a server"
  "git:A distributed version control system"
  "vim:A highly configurable text editor"
  "zsh:A shell designed for interactive use"
  "jq:A lightweight and flexible command-line JSON processor"
  "tree:A recursive directory listing command"
  "ncdu:A disk usage analyzer with an ncurses interface"
  "glances:A cross-platform system monitoring tool"
  "tldr:A collection of simplified and community-driven man pages"
  "neofetch:A command-line system information tool"
  "bat:A cat(1) clone with wings"
  "fd-find:A simple, fast and user-friendly alternative to 'find'"
  "ripgrep:A line-oriented search tool that recursively searches your current directory for a regex pattern"
  "fish:Smart and user-friendly command line shell"
  "emacs-nox:Extensible, customizable, free/libre text editor (no X version)"
  "lynx:A text-based web browser"
  "httrack:A free and easy-to-use offline browser utility"
  "silversearcher-ag:A code-searching tool similar to ack, but faster"
  "httpie:A user-friendly command-line HTTP client for the API era"
  "mtr:Network diagnostic tool"
  "iftop:Display bandwidth usage on an interface"
  "tcpdump:Command-line packet analyzer"
  "whois:Client for the whois directory service"
  "lsof:Utility to list open files"
  "strace:Trace system calls and signals"
  "sysstat:System performance tools for Linux"
  "dstat:Versatile resource statistics tool"
  "dnsutils:DNS utilities provided by BIND"
  "net-tools:Networking utilities"
)

# Loop through each package and install it
for package in "${packages[@]}"; do
    # Split the package name and description
    IFS=":" read -r name description <<< "$package"

    # Check if the package is already installed or not
    if dpkg -l | grep -q "^ii  $name "; then
        echo "$name ($description) is already installed"
    else
        # Try to install the package and print a message based on the result
        if sudo apt install -y "$name"; then
            echo "$name ($description) installed successfully"
        else
            echo "Failed to install $name ($description)"
            exit 1
        fi
    fi
done

# Worked on Debian 12 (stable); should work on Ubuntu and other Debian-based distributions as well.
