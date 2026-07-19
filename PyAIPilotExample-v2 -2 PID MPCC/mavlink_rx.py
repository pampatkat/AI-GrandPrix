import struct
import time
import threading

from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2

class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}
        self.last_imu_status_time = 0.0

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon = False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                # Simulator restarts briefly close the UDP endpoint on Windows.
                time.sleep(0.01)
                continue

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            elif msg_type == "COMMAND_ACK":
                self.on_command_ack(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            #
            #
            # PLEASE NOTE:
            # As per the configuration of the latest version of the simulator, Attitude telemetry has been disabled.
            #
            #
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            #
            #
            # PLEASE NOTE:
            # As per the configuration of the latest version of the simulator, Local Position NED telemetry has been disabled.
            #
            #
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            #
            #
            # PLEASE NOTE:
            # As per the configuration of the latest version of the simulator, Odometry telemetry has been disabled.
            #
            #
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

    def on_heartbeat(self, msg):
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self.data['heartbeat'] = {
            'armed': armed,
            'base_mode': msg.base_mode,
            'custom_mode': msg.custom_mode,
            'system_status': msg.system_status,
            'time': time.time(),
        }

    def on_command_ack(self, msg):
        self.data['last_command_ack'] = {
            'command': msg.command,
            'result': msg.result,
            'time': time.time(),
        }

    def on_timesync(self, msg):
        received_ns = time.time_ns()
        last_request_ns = self.data.get("timesync_last_request_ns")
        if last_request_ns is None:
            return
        fields = (int(msg.tc1), int(msg.ts1))
        client_time = None
        server_time = None
        echoed_fields = [
            (abs(value - last_request_ns), index, value)
            for index, value in enumerate(fields)
            if value > 0
        ]
        if echoed_fields:
            difference_ns, index, value = min(echoed_fields)
            if difference_ns < 1_000_000_000:
                client_time = value
                server_time = fields[1 - index]
        if client_time is None or server_time is None or server_time < 1_000_000_000_000_000:
            return
        round_trip_ns = received_ns - client_time
        if not 0 <= round_trip_ns <= 200_000_000:
            return
        offset_ns = ((client_time + received_ns) // 2) - server_time
        previous = (self.data.get("timesync") or {}).get("server_to_local_offset_ns")
        if previous is not None:
            offset_ns = int(0.9 * previous + 0.1 * offset_ns)
        self.data["timesync"] = {
            "server_to_local_offset_ns": offset_ns,
            "round_trip_ms": round_trip_ns / 1e6,
            "time": time.time(),
        }

    def on_attitude(self, msg):
        #
        #
        # PLEASE NOTE:
        # As per the configuration of the latest version of the simulator, Attitude telemetry has been disabled.
        #
        #
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        roll_speed = msg.rollspeed
        pitch_speed = msg.pitchspeed
        yaw_speed = msg.yawspeed
        time_boot_ms = msg.time_boot_ms

    def on_local_position_ned(self, msg):
        #
        #
        # PLEASE NOTE:
        # As per the configuration of the latest version of the simulator, Local Position NED telemetry has been disabled.
        #
        #
        pos_x = msg.x
        pos_y = msg.y
        pos_z = msg.z
        vel_x = msg.vx
        vel_y = msg.vy
        vel_z = msg.vz
        time_boot_ms = msg.time_boot_ms
        self.data['local_position'] = {'x': pos_x, 'y': pos_y, 'z': pos_z}
        self.data['local_velocity'] = {'x': vel_x, 'y': vel_y, 'z': vel_z}

    def on_odometry(self, msg):
        #
        #
        # PLEASE NOTE:
        # As per the configuration of the latest version of the simulator, Odometry telemetry has been disabled.
        #
        #
        pos_x, pos_y, pos_z = msg.x, msg.y, msg.z
        qx, qy, qz, qw = msg.q[1], msg.q[2], msg.q[3], msg.q[0]
        vel_x, vel_y, vel_z = msg.vx, msg.vy, msg.vz
        roll_speed = msg.rollspeed
        pitch_speed = msg.pitchspeed
        yaw_speed = msg.yawspeed
        time_boot_us = msg.time_usec
        reset_count = msg.reset_counter
        self.data['local_position'] = {'x': pos_x, 'y': pos_y, 'z': pos_z}
        self.data['local_velocity'] = {'x': vel_x, 'y': vel_y, 'z': vel_z}

    def on_highres_imu(self, msg):
        acceleration_x, acceleration_y, acceleration_z = msg.xacc, msg.yacc, msg.zacc
        gyro_x, gyro_y, gyro_z = msg.xgyro, msg.ygyro, msg.zgyro
        time_boot_us = msg.time_usec
        received_ns = time.time_ns()
        timestamp_ns = received_ns
        timestamp_source = "receive_time"
        if time_boot_us >= 1_000_000_000_000:
            offset_ns = (self.data.get("timesync") or {}).get(
                "server_to_local_offset_ns", 0
            )
            timestamp_ns = int(time_boot_us) * 1000 + int(offset_ns)
            timestamp_source = "imu_epoch"
        else:
            boot_epoch_ns = self.data.get("sim_boot_epoch_ns")
            if boot_epoch_ns is not None:
                timestamp_ns = int(boot_epoch_ns + int(time_boot_us) * 1000)
                timestamp_source = "imu_boot_aligned"
        imu_buffer = self.data.get("imu_buffer")
        if imu_buffer is not None:
            imu_buffer.add(
                timestamp_ns / 1e9,
                [acceleration_x, acceleration_y, acceleration_z],
                [gyro_x, gyro_y, gyro_z],
                timestamp_source,
            )
            if time.time() - self.last_imu_status_time >= 0.25:
                self.last_imu_status_time = time.time()
                self.data["imu_status"] = imu_buffer.stats(received_ns / 1e9)

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        # data_type - ID of this message
        # sim_boot_time_ms - elapsed ms on server since sim boot
        # race_start_boot_time_ms - elapsed ms on server since sim boot when race started. None or < 0 if race has not started
        # race_finish_time_ns - elapsed ns on server since sim boot when race finished. None or < 0 if race is ongoing
        # active_gate_index - current index of target race gate
        # last_gate_race_time - race time in seconds when last gate was passed
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload)
        boot_epoch_candidate = time.time_ns() - int(sim_boot_time_ms) * 1_000_000
        previous_boot_epoch = self.data.get("sim_boot_epoch_ns")
        if previous_boot_epoch is None or abs(
            boot_epoch_candidate - previous_boot_epoch
        ) > 1_000_000_000:
            self.data["sim_boot_epoch_ns"] = boot_epoch_candidate
        else:
            self.data["sim_boot_epoch_ns"] = int(
                0.98 * previous_boot_epoch + 0.02 * boot_epoch_candidate
            )
        self.data['race_status'] = {
            'sim_boot_time_ms': sim_boot_time_ms,
            'race_start_boot_time_ms': race_start_boot_time_ms,
            'race_finish_time_ns': race_finish_time_ns,
            'active_gate_index': active_gate_index,
            'last_gate_race_time': last_gate_race_time,
            'time': time.time(),
        }

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload = full_payload + self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        #
        #
        # PLEASE NOTE:
        # As per the configuration of the latest version of the simulator, gate positions, orientations and dimensions are no longer published in telemetry and will be nulled.
        #
        #
        # header:
        #   num_gates - track gate count
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        for i in range(num_gates):
            # Gate Info
            #   gate_id - range is 0 - num_gates
            #   position_ned_x, position_ned_y, position_ned_z - Position of gate in NED coordinates
            #   orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z - Orientation of gate in NED coordinates
            #   width - gate width in metres
            #   height - gate height in metres
            gate_id, position_ned_x, position_ned_y, position_ned_z, orientation_ned_w, orientation_ned_x, orientation_ned_y, orientation_ned_z, width, height = struct.unpack_from(
                "<Hfffffffff", payload)
            payload = payload[38:]

    def on_actuator_output_status(self, msg):
        time_boot_us = msg.time_usec
        motor_front_left = msg.actuator[0]
        motor_front_right = msg.actuator[1]
        motor_back_left = msg.actuator[2]
        motor_back_right = msg.actuator[3]

    def on_collision(self, msg):
        # Collision IDs
        # 1001 - Gate
        # 1002 - Environment
        collision_id = msg.id

        threat_level = msg.threat_level # 1-2 with 2 being higher impact collision
        impact = msg.horizontal_minimum_delta # this is not a delta - it is the impulse magnitude in kg m/s
