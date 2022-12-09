FROM continuumio/miniconda3

# install some dependencies
RUN apt-get update --fix-missing \
	&& apt-get install -y \
		ca-certificates \
		libglib2.0-0 \
	 	libxext6 \
	   	libsm6  \
	   	libxrender1 \
		libxml2-dev \
		gcc \
		make \
		pandoc \
		pandoc-citeproc

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt install -y ./google-chrome-stable_current_amd64.deb

# Install base R and devtools
RUN apt-get install wget r-base r-base-dev -y 

# install python3 & virtualenv
RUN apt-get install -y \
		python3-pip \
		python3-dev 
RUN apt-get install -y chromium-driver
  
RUN export PATH=~/anaconda3/bin:$PATH
# install conda dependencies
RUN conda config --add channels conda-forge
RUN conda install -c conda-forge -y pip \
        altair==4.2.0 \
        pandas==1.4.3 \
        numpy==1.23.3 \
        ipykernel \
        ipywidgets \
        'ipython>=7.15' \
        vega_datasets \
        altair_saver \
        'selenium<4.3.0' \
	scikit-learn \
	matplotlib

# install pip dependencies
RUN pip install --upgrade pip \
        docopt-ng==0.8.1 \
        webdriver-manager \
        dataframe_image==0.1.1 \
        jupyter_core \
        jupyter_client \
	lxml \
        && rm -fr /root/.cache

ENV LD_LIBRARY_PATH /usr/local/lib/R/lib/:${LD_LIBRARY_PATH}

# install R packages
RUN Rscript -e "install.packages('knitr')" 
RUN Rscript -e "install.packages('rmarkdown')"
RUN Rscript -e "install.packages('kableExtra')"
RUN Rscript -e "install.packages('tidyverse')"

