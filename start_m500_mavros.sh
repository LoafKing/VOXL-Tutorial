#!/bin/bash

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash 2>/dev/null

echo "[1/2] starting roscore..."
gnome-terminal -- bash -lc "source /opt/ros/noetic/setup.bash; roscore; exec bash"

sleep 3

echo "[2/2] starting mavros..."
gnome-terminal -- bash -lc "source /opt/ros/noetic/setup.bash; source ~/catkin_ws/devel/setup.bash 2>/dev/null; roslaunch mavros px4.launch fcu_url:='udp://:14550@192.168.8.1:14550'; exec bash"
