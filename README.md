# VOXL-Tutorial
Repository for VOXL configuration and test.

The details of ModalAI VOXL Flight PX4 can be found in https://docs.ncnynl.com/en/px4/en/flight_controller/modalai_voxl_flight.html

QGroundControl version: 3.4.4

Use QGcontrol to do calibaration and RC control configuration. VOXL Modal Portal can be opened with http://<VOXL IP>/.

To enter the VOXL embedded system Linux, connect the drone USB connector J8 with your PC:

############

adb devices

adb shell

bash

############

Or you can connect to the VOXL-Test wifi, using m500 ssh info to enter the onboard system with the of drone's ip address.
