import json
import threading
import time
from typing import Dict, List

from serial import Serial, SerialException

IDLE_POWER = 180.5  # Watts
NUM_FIELDS = 18
POWER_FIELDS = ["P1", "P2", "P3", "P4", "P5", "P6"]


class PowerMonitor(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, stop_event, *args, **kwargs):
        super(PowerMonitor, self).__init__(*args, **kwargs)
        self._stop_event = stop_event

    def stop(self):
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @staticmethod
    def try_open_serial_port(serial_port: str = "/dev/ttyUSB0", baud_rate: int = 115200) -> bool:
        try:
            return Serial(serial_port, baud_rate)
        except:
            return None

    @staticmethod
    def try_close_serial_port(ser: Serial):
        try:
            ser.close()
        except:
            pass

    @staticmethod
    def read_power_monitor(ser: Serial, stop_event: threading.Event, power_readings: List[Dict]):
        data_str = ""

        def _wrap_up():
            ser.close()
            outputs = data_str.strip().split("\n")
            for output in outputs:
                try:
                    power_readings.append(json.loads(output))
                except json.JSONDecodeError:
                    continue

                # content = output.strip().split(",")
                # if len(content) != NUM_FIELDS:
                #     continue
                # power_readings.append({x.split(":")[0]: x.split(":")[1] for x in content})

        while not stop_event.is_set():
            try:
                # Check if incoming bytes are waiting to be read from the serial input
                # buffer.
                if ser.in_waiting > 0:
                    data_str += ser.read(ser.in_waiting).decode("utf-8")
            except UnicodeDecodeError as e:
                print(e, "Decode error, skip")
                continue
            except SerialException as e:
                print(e, "SerialException, skip")
                continue
            except:
                break
            time.sleep(0.1)
        _wrap_up()


def main():
    ser = PowerMonitor.try_open_serial_port()
    if ser is not None:
        results = []
        stop_event = threading.Event()
        power_monitor = PowerMonitor(
            stop_event=stop_event, target=PowerMonitor.read_power_monitor, args=(ser, stop_event, results)
        )
        power_monitor.start()
    else:
        print("failed to open serial port")
        exit()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        power_monitor.stop()
        power_monitor.join()
        PowerMonitor.try_close_serial_port(ser)
        for r in results:
            print(r)


if __name__ == "__main__":
    # TODO: address multiple reads
    main()
