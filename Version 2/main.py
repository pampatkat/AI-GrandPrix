#
# Sample Python client for the AI GP controller
#

import time

from setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']

# the client stays running across races: it re-arms and re-runs the countdown
# automatically every time the simulator schedules a new race or is reset, so
# you can leave this running alongside the simulator instead of restarting it.
print("Client ready - running alongside the simulator. (press Ctrl+C to stop)", flush=True)
is_running = True
try:
    while is_running:
        print("Arming drone...", flush=True)
        controller.arm()

        # countdown begins automatically once the simulator schedules a race.
        # the drone hovers in place through "3... 2... 1..." then flies forward
        # on "GO!"
        print("Get ready...", flush=True)
        controller.run_countdown()

        # fly forward until the simulator is reset or a new race starts, then
        # loop back to re-arm and run the countdown again.
        print("Flying forward! (reset the simulator to run again, Ctrl+C to stop)", flush=True)
        controller.fly_until_reset()

        print("Simulator reset detected - getting ready for the next run...\n", flush=True)
except KeyboardInterrupt:
    print("\nStopping (Ctrl+C received)...", flush=True)

# graceful shutdown
print("Disarming drone...", flush=True)
controller.disarm()

ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
