# Starter Code Notes

setup.py
- setup_components(shared_data, system_boot_ms, server_ip, server_udp_port)
    - Initializes components for the main program
    - Opens MAVLink UDP connection to simulator
    - Creates MAVLink receiver, timesync loop, vision receiver, controller

main.py
- SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    - Simulator UDP endpoint for MAVLink
- system_boot_ms
    - Local boot timestamp in ms
- shared_data
    - Shared runtime dictionary for components
- components
    - Dict returned by `setup_components`
- controller, ts_loop, mavlink_rx, vision_rx
    - Component references used by the main loop
- is_running
    - Main loop flag
- Behavior:
    - Starts the controller, arms the drone, enters update loop
    - Joins component threads on exit

controller.py
- MAVLINK_CMD_SIM_RESET
    - Custom MAVLink reset command ID
- MOTOR_FRONT_LEFT, MOTOR_FRONT_RIGHT, MOTOR_BACK_LEFT, MOTOR_BACK_RIGHT
    - Motor output values for actuator control
- PITCH_RATE, ROLL_RATE, YAW_RATE, THRUST
    - Fixed attitude/rate control targets
- RATES_ATTITUDE_MASK
    - MAVLink attitude ignore mask
- VELOCITY_POSITION_MASK
    - MAVLink position target ignore mask
- CONTROL_HZ
    - Control loop frequency
- update_motor_control(mavlink_conn, system_boot_ms)
    - Sends actuator control target with motor RPM values
- update_attitude_flight_control(mavlink_conn, system_boot_ms)
    - Sends desired attitude target message
- update_position_flight_control(mavlink_conn, system_boot_ms)
    - Sends local NED velocity position target message
- Controller
    - __init__(sim_conn, data, system_boot_ms)
    - update()
        - Calls `update_motor_control` and sleeps to maintain control rate
    - arm()
        - Sends arm command to the vehicle
    - send_sim_reset_command()
        - Sends simulator reset command

mavlink_rx.py
- ENCAPSULATED_RACE_STATUS_MSG_ID, ENCAPSULATED_TRACK_INFO_MSG_ID
    - Payload type identifiers for custom encapsulated data
- MAVLinkRX
    - __init__(mavlink_connection, data)
    - create_mavlink_rx(cls, mavlink_connection, data)
        - Starts MAVLink receive thread
    - get_thread_for_join()
        - Stops receiver loop and returns thread for join
    - mavlink_receive_loop()
        - Receives MAVLink messages and dispatches by type
    - on_heartbeat(msg)
        - Parses heartbeat armed state
    - on_timesync(msg)
        - Parses timesync payload fields
    - on_attitude(msg)
        - Parses attitude Euler and rates
    - on_local_position_ned(msg)
        - Parses local position and velocity
    - on_odometry(msg)
        - Parses odometry pose, velocity, and reset count
    - on_highres_imu(msg)
        - Parses IMU acceleration and gyro values
    - on_encapsulated_data(msg)
        - Routes custom encapsulated messages
    - on_race_status(msg)
        - Unpacks race status packet
    - on_track_data_packet(msg)
        - Reassembles chunked track data packets
    - on_track_data(payload)
        - Parses complete track gate payload
    - on_actuator_output_status(msg)
        - Parses actuator motor outputs
    - on_collision(msg)
        - Parses collision event and threat data

vision_rx.py
- SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    - UDP endpoint for receiving vision frame packets
- VisionRX
    - __init__(data)
        - Starts UDP receive thread
    - get_thread_for_join()
        - Stops vision loop and returns thread
    - _vision_loop()
        - Receives UDP packets, reassembles JPEG frames, decodes images
    - process_frame(frame_id, img)
        - Placeholder for FPV image processing

timesync.py
- TIMESYNC_REQUEST_HZ
    - Frequency for timesync requests
- TimeSync
    - __init__(mavlink_connection, data)
    - create_timesync(cls, mavlink_connection, data)
        - Starts timesync thread
    - get_thread_for_join()
        - Stops timesync loop and returns thread
    - timesync_loop()
        - Periodically sends MAVLink TIMESYNC requests

