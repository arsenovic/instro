import time
import serial
import numpy as np
from collections.abc import Sequence
import skrf as skrf
from instro.unstable.vna.vna import VNADriverBase

class NanoVNAv1(VNADriverBase):
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, timeout=3.0):
        """Initializes serial connection to the text-based NanoVNA firmware."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.open()

    def open(self):
        """Opens the serial port interface."""
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        time.sleep(0.5) # Allow connection to settle
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self._is_open=True

    def close(self):
        """Closes the serial port interface safely."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self._is_open =False

    def _send_command(self, cmd_string):
        """Sends an ASCII text command and flushes buffers."""
        full_cmd = f"{cmd_string}\r\n".encode('ascii')
        self.ser.write(full_cmd)
        self.ser.flush()
        time.sleep(0.1) # Small processing delay for the VNA micro-controller

    def _read_lines(self):
        """Reads back incoming lines until the trailing prompt or timeout occurs."""
        lines = []
        start_time = time.time()
        while (time.time() - start_time) < self.timeout:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                # If we encounter the command prompt character, parsing is complete
                if 'ch0>' in line or 'ch1>' in line or '>' in line:
                    break
                lines.append(line)
        return lines

    def get_f(self):
        """Queries the device for the active sweep frequencies."""
        self.ser.reset_input_buffer()
        self._send_command("frequencies")
        raw_lines = self._read_lines()
        
        freqs = []
        for line in raw_lines:
            try:
                # Filter out echoed echo text lines or system strings
                if "frequencies" in line:
                    continue
                val = float(line.strip())
                freqs.append(val)
            except ValueError:
                continue
                
        return np.array(freqs)

    def get_channel_data(self, array_index=0):
        """Fetches raw S-parameter values (array 0 = S11, array 1 = S21)."""
        self.ser.reset_input_buffer()
        self._send_command(f"data {array_index}")
        raw_lines = self._read_lines()
        
        complex_data = []
        for line in raw_lines:
            try:
                if f"data {array_index}" in line:
                    continue
                # V1 formats raw real/imaginary values split by whitespace
                parts = line.strip().split()
                if len(parts) >= 2:
                    real = float(parts[0])
                    imag = float(parts[1])
                    complex_data.append(complex(real, imag))
            except ValueError:
                continue
                
        return np.array(complex_data)


    
    def get_frequency(self, cnum = None):
        return skrf.Frequency.from_f(self.get_f(), unit='hz')

    def set_frequency(self, freq, cnum = None):
        raise NotImplementedError()

    def get_sparam(self, m, n, cnum = None):
        if (m,n)==(0,0):
            self.get_channel_data(array_index=0)
        elif (m,n) ==(1,0):
            self.get_channel_data(array_index=10)
        else:
            raise ValueError('unsupported s-parameter index: S{m}{n}')

    def get_snp_network(
            self,  
            ports: Sequence[int | None], 
            cnum: int | None = None,
            **kw
            ) -> skrf.Network:
        
        frequency = self.get_frequency()
        frequency.unit = 'ghz'

        if ports == [0]:
            s = self.get_channel_data(array_index=0)

        elif ports == [0,1] or ports ==None:
            s11 = self.get_channel_data(array_index=0)
            s21 = self.get_channel_data(array_index=1)
            s = np.zeros((len(s11), 2, 2), dtype=complex)
            s[:, 0, 0] = s11  # S11
            s[:, 1, 0] = s21  # S21
            s[:, 0, 1] = 0.0  # S12 (Not hardware supported)
            s[:, 1, 1] = 0.0  # S22 (Not hardware supported)
        else:
            ValueError('nanovna only supports the following ports values: [0], [0,1]')
        
        network = skrf.Network(frequency=frequency , s=s,**kw )
        return network

 