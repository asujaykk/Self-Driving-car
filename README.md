# Self-Driving-Car (simulator version)
This project demonstrate basic self driving car model using udacity car driving simulator

# Steps involved
  1. Setting up the environment. 
  2. Setting up udacity driving simulator.
  3. Creating training data from the simulator
  4. Build and train the model with the training data.
  5. Testing the model using behavior cloning project
  
## 1. Setting up the environment.
   First create a prject folder structure.   
   
   Create 'autopilot_project' folder for keeping all required files and modules for this projects.
           
           mkdir autopilot_project 
   Under this folder create a 'data_set/train_data' , 'data_set/test_data'  folder for storing train and test driving data. 
           
           cd autopilot_project  
           
           mkdir data_set/train_data   # for storing traing driving data 
           
           mkdir data_set/test_data    # for storing validation drivng data  
          
   Then clone this repository to 'autopilot_project' folder with below command.  
    
          git clone https://github.com/asujaykk/Self-Driving-car.git 
    
   Then clone UDACITY behavioural cloning repository to this folder (for finally testing your model)   
          
          git clone https://github.com/udacity/CarND-Behavioral-Cloning-P3.git 
    
## 2. Setting up udacity driving simulator.    
   
   Fist download udacity prebuild term1 driving simultor from this link, 
   
           wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip
           
   If you want to get different versions or want to build custom simulator, then please check UDACITY self-driving-car-sim repository here https://github.com/udacity/self-driving-car-sim
   
   After downloading please extract 'term1-simulator-linux.zip' file to the current directory.
   
##  3. Creating training data from the simulator
   
   
