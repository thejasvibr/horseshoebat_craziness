# horseshoebat_craziness
A bunch of modules related to the horseshoe bat craziness project


# Setting up python 
1) Download Anaconda 
2) Remember to add conda to your OS path
3) Open Anaconda Prompt. Create a new environment to avoid package compatibility issues, here we'll call the environment 'horseshoebat_craziness'

  *conda create --name horseshoebat_craziness python=2.7*
  
4) Activate the environment 

  *activate horseshoebat_craziness*
  
5) Install the following required packages:

*conda install scipy numpy matplotlib pandas*

6) Include a connection between the environment and jupyter notebook:

  *conda install ipykernel*
  
7) Download the peakutils package from here https://bitbucket.org/lucashnegri/peakutils/downloads/
8) Unzip it, change current directory in Anaconda prompt to the unzipped folder 
9) Install the peakutils package with the following command : 

  *python setup.py install*
10) all required packages should be there now. 
