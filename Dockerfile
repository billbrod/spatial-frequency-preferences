FROM mambaorg/micromamba

# git is necessary for one of the packages we install via pip
# and we need to be root to use apt
USER root
RUN apt -y update
RUN apt -y install git

USER micromamba

RUN mkdir -p /home/sfp_user/spatial-frequency-preferences
COPY . /home/sfp_user/spatial-frequency-preferences

RUN micromamba install -n base -y -f /home/sfp_user/spatial-frequency-preferences/environment.yml && \
    micromamba clean --all --yes

# USER sfp_user
