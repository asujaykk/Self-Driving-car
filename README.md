# Self-Driving-Car (simulator version)
This project demonstrate basic self driving car model using udacity car driving simulator. In this project we will be building a Convolution Neural Network model to predict the steering angle for a virtual car in the simulator running at a constat speed.The goal is to drive the car in the simulator autonomously for a full lap without deviating from the main track/road. 

1. As a first step, we will show the model(training) how to drive in the track by manually driving the car (simulator in training mode) without making any mistake.
2. During this training drive, we will record the vehicle sensor values (virtual camera images of the car, steering angle sensor value, speed sensor value) in to a folder.
3. Then we will feed this data to a CNN model for learning (Actual model training). During this training time the model will learn the steering angle or the maneuvers that you applied for different road situations like left curve, straight road, right curve , approaching the side lane and departing from the side lane. 
4. After training the model, we will run the model with real time virtual camera sensor data from the simulator as input , and use the output of the model to control the steering angle of the car in the simulator, in effect the model will be controling the steering of the car in the simulator.


Udacity self driving car simulator is used testing and training our model.

# Steps involved
  1. Setting up the environment. 
  2. Setting up udacity driving simulator.
  3. Creating training/test data from the simulator.
  4. Build and train the model with the training data.
  5. Testing the model using Behavioral-Cloning project.
  
## 1. Setting up the environment.
   First create a project folder structure.   
   
   Create 'autopilot_project' folder for keeping all required files and modules for this project.
           
           mkdir autopilot_project 
   Under this folder create a 'data_set/train_data' , 'data_set/test_data'  folder for storing training and test driving data. 
           
           cd autopilot_project  
           
           mkdir -p data_set/train_data   # for storing training driving data 
           
           mkdir -p data_set/test_data    # for storing validation drivng data  
          
   Then clone this repository to 'autopilot_project' folder with below command.  
    
          git clone https://github.com/asujaykk/Self-Driving-car.git 
    
   Then clone UDACITY 'CarND-Behavioral-Cloning-P3' repository to this folder (for finally testing your model)   
          
          git clone https://github.com/udacity/CarND-Behavioral-Cloning-P3.git 
          
   Also create an anaconda virtual environment 'tensgpu_1' with the 'autopilot_project/Self-Driving-car/anaconda_env/tens_gpu_self_driving_car.yaml'  file for training and testing the keras model.  
   
          conda env create -f  autopilot_project/Self-Driving-car/anaconda_env/tens_gpu_self_driving_car.yaml 
    
## 2. Setting up udacity driving simulator.    
   
   Download udacity pre-build term1 driving simultor with below command (term1 beta simulator). 
   
           wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip
           
   To get different versions or to build a custom simulator, then please check  UDACITY self-driving-car-sim repository here https://github.com/udacity/self-driving-car-sim
   
   After downloading please extract 'term1-simulator-linux.zip' file to 'autopilot_project' folder.
   Open terminal and navigate to 'autopilot_project/term1-simulator-linux/beta_simulator_linux' and make 'beta_simulator.x86_64' (or beta_simulator.x86 based on your system architecture) as executable with the following command.  
   
           sudo chmod +x beta_simulator.x86_64
   
   
##  3. Creating training/test data from the simulator.
   
   1. First run the pre-build simulator executable from the extracted folder.
   2. Once it is launched, choose training option from the main window.
   3. Now click on the record button and choose 'data_set/train_data' folder for saving training data.
   4. Align the car on the track and then click on the record button again, then drive the car in the track for 7 or 8 laps. Click on the record button again to stop recording.
   5. Restart the simulator and  click on the record button and choose 'data_set/test_data' folder for saving test data.
   6. And repeat step 4 for 2 or 3 laps.
   7. After recording, the recorded images and driving_log.csv files can be found under respective folders.

## 4. Build and train the model with the training data.
   1. Change current working directory to 'autopilot_project/Self-Driving-car'
   2. Activate the anaconda environmnet 'tensgpu_1' that was created before using below command.

             conda activate tensgpu_1
   3. Then run 'model_train.py' to create a model and train it with the training and test data set that was created before. 
       
             python3 model_train.py --train_csv_file 'path to training driving_log.csv file' --test_csv_file   'path to test driving_log.csv file' --batch_size 32 --epochs 50  1>train.log 2>&1
             
      The above execution will create four different models (for diffrenet learning rates) under the folder 'autopilot_project/Self-Driving-car/models'. Check 'autopilot_project/Self-Driving-car/train.log' to see the progress of training. 
      After successsfull training, revisit the log and check which model had minimum  'loss' and 'val_loss', and choose that as final model for testing.
      
      Note: If your PC have resource constraints then please reduce batchsize to 8 or 16 to avoid 'OOM' error.
      
             
##  5. Testing the model using Behavioral-Cloning project.

   1. Lauch the simulator in autonomous mode For testing the model.
   2. Run the pre-build simulator executable.
   3. Once it is launched, choose Autonomous mode from the main window (Now the simulator should be ready to accept a connection).
   4. Change the working directory to 'autopilot_project/CarND-Behavioral-Cloning-P3'
   5. Activate the anaconda environmnet 'tensgpu_1'.

             conda activate tensgpu_1
   6. Then run 'drive.py' with the folllwoing command.
   
             python3 drive.py 'path to the craeted model.h5 file'
        Note: If you are facing any issues then please check issues under 'https://github.com/udacity/CarND-Behavioral-Cloning-P3' repo to get solutions.  
   7. If the environment is proper and if the script able to make a connection with the simulator then the car in the simulator start running at 9kmph, and  it will try to adjust it's steering angle to keep the car always on the track. 
   8. If the car is able to maintain on the track for a full lap, then your model is performing well :)
   10. If The car is not always stays on the track, then the model is poorly performing :( , then retrain the model with more data and with reduced batch size. and test it again and agian until a good performance is achived.
   
   11. I also kept a trained model  'save_at_8.h5' at https://drive.google.com/file/d/1VkyFqVZIGY8Oayi_i3R4czwdgNHZKxxt/view. This model will work perfectly with term1-beta simulator track1. The model performed well  even with max speed of 30Mph.
   
       The following GIF shows the output of 'save_at_8.h5' model. 
       
       ![20221112_015409](https://user-images.githubusercontent.com/78997596/201426129-31a1f8b6-6f5f-4655-a493-720745345d70.gif)

   


References:
  1. https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
  2. https://github.com/udacity/self-driving-car-sim
  
       
    
