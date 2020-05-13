#!/bin/bash

echo Testing steering
timeout 10s rostopic echo /vehicle/steering_cmd > test_steering_cmd.out
cat test_steering_cmd.out | grep '\-\-\-' | wc -l

echo Testing braking
timeout 10s rostopic echo /vehicle/brake_cmd > test_brake_cmd.out
cat test_brake_cmd.out | grep '\-\-\-' | wc -l

echo Testing throttle
timeout 10s rostopic echo /vehicle/throttle_cmd > test_throttle_cmd.out
cat test_throttle_cmd.out | grep '\-\-\-' | wc -l
