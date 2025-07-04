#!/bin/bash                                                                                                                                                                                                          
                                                                                                                                                                                                                     
#SBATCH --job-name=mass_maps                                                                                                                                                                                  
#SBATCH --error=Mass_maps.err                                                                                                                                                                                 
#SBATCH --output=Mass_maps.out                                                                                                                                                                                
#SBATCH --clusters=cm4                                                                                                                                                                                               
#SBATCH --partition=cm4_tiny                                                                                                                                                                                         
#SBATCH --qos=cm4_tiny                                                                                                                                                                                               
#SBATCH --get-user-env                                                                                                                                                                                               
#SBATCH --nodes=1                                                                                                                                                                                                    
#SBATCH --ntasks-per-node=8                                                                                                                                                                                         
#SBATCH --cpus-per-task=14                                                                                                                                                                                         


export OMPI_MCA_btl=^openib

mpiexec -np 8 python get_maps_magneticum.py config.yaml
