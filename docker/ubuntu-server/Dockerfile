##
# atmoswing/atmoswing:ubuntu-server

FROM ubuntu:18.04 AS builder

MAINTAINER Pascal Horton <pascal.horton@alumnil.unil.ch>

# Add repo for Jasper
RUN add-apt-repository -y "deb http://security.ubuntu.com/ubuntu xenial-security main"

# Update
RUN apt update -y

# Setup build environment
RUN apt install -y git build-essential gnome-devel cmake make wget unzip

# Install libraries
RUN apt install -y libnetcdf-dev libproj-dev libgdal-dev libcurl3 libjasper-dev \
    libssl-dev libwxbase3.0 libwxbase3.0-dev libeccodes-dev libjasper1

