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
          
   Also create an anaconda virtual environment with the 'yaml'  file for training and testing your keras model.  
   
          conda -env -m yaml   
    
## 2. Setting up udacity driving simulator.    
   
   Fist download udacity prebuild term1 driving simultor from this link, 
   
           wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip
           
   If you want to get different versions or want to build custom simulator, then please check releases of UDACITY self-driving-car-sim repository here https://github.com/udacity/self-driving-car-sim
   
   After downloading please extract 'term1-simulator-linux.zip' file to the current directory.
   
   
##  3. Creating training data from the simulator
   
   1. First run the prebuild simulator executable from the extracted folder.
   2. Once it is launched, choose training option from the main window.
   3. Now click on the record button and choose 'data_set/train_data' folder for recording training data.
   4. Align the car on the track and then click on the record button, then drive the car in the lap for 7 or 8 laps and then click on the record button again to stop recording
   5. Now again click on the record button and choose 'data_set/test_data' folder for recording test data.
   6. And repeat step 4 for 2 or 3 laps.
   7. After recording, you can see the recorded images and driving_log.csv files under respective folders.

## 4. Build and train the model with the training data.
   1. Change your cwd to 'autopilot_project/Self-Driving-car'
   2. activate the anaconda environmnet that you created earlier using below command.

             cond activate 'env_name'
   3. Then run 'model_train.py' to create a model and train it with the training and test data set that we created earlier. 
       
             python3 model_train.py --train_csv_file 'path to training driving_log.csv file' --test_csv_file   'path to test driving_log.csv file'
             
   4. 
