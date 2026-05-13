#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import time
import wave
from typing import List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from scipy.signal import butter, filtfilt

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

try:
    import pyaudio
except Exception:
    pyaudio = None


def bandpass_filter(data, low=50, high=60, fs=500):
    b, a = butter(N=4, Wn=[low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, data)


def highpass_filter(data, cutoff=10, fs=500):
    b, a = butter(N=4, Wn=cutoff / (fs / 2), btype="highpass")
    return filtfilt(b, a, data)


def save_file_name(basename, outdir, ext):
    idx = 0
    os.makedirs(outdir, exist_ok=True)
    while os.path.exists(os.path.join(outdir, f"{basename}_{idx}.{ext}")):
        idx += 1
    return os.path.join(outdir, f"{basename}_{idx}.{ext}")


def _safe_channels(getter_fn, board_id) -> List[int]:
    try:
        ch = getter_fn(board_id)
        return list(ch) if ch is not None else []
    except Exception:
        return []


def _safe_channel(getter_fn, board_id) -> Optional[int]:
    try:
        ch = getter_fn(board_id)
        return int(ch) if ch is not None else None
    except Exception:
        return None


def _maybe_filter_2d(x_2d: np.ndarray, do_filter: bool, fs: int, min_len: int = 128) -> np.ndarray:
    if not do_filter:
        return x_2d
    if x_2d is None or x_2d.size == 0:
        return x_2d

    x = x_2d if x_2d.ndim == 2 else np.atleast_2d(x_2d)
    n = x.shape[1]
    if n < int(min_len):
        return x

    try:
        y = x.copy()
        y = highpass_filter(y, cutoff=10, fs=fs)
        y = bandpass_filter(y, low=50, high=60, fs=fs)
        return y
    except Exception:
        return x


class ContinuousAudioRecorder:
    def __init__(
        self,
        out_dir: str,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk: int = 1024,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.wav_path = save_file_name("audio", self.out_dir, "wav")
        self.info_csv_path = save_file_name("audio_info", self.out_dir, "csv")

        self.device_index = device_index
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk = int(chunk)

        self.thread = None
        self.stop_event = None

        self.start_unix_time_s: Optional[float] = None
        self.sample_width: Optional[int] = None

    def _write_info_csv(self):
        if self.start_unix_time_s is None or self.sample_width is None:
            return

        utc = pd.to_datetime(self.start_unix_time_s, unit="s", utc=True)
        local = utc.tz_convert("America/New_York")

        with open(self.info_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "audio_start_unix_time_s",
                "audio_start_utc_iso",
                "audio_start_local_iso",
                "sample_rate_hz",
                "channels",
                "sample_width_bytes",
                "wav_path",
            ])
            writer.writerow([
                f"{self.start_unix_time_s:.6f}",
                utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                local.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                self.sample_rate,
                self.channels,
                self.sample_width,
                self.wav_path,
            ])

    def _run(self):
        if pyaudio is None:
            print("[Audio] pyaudio not installed; audio disabled.")
            return

        pa = pyaudio.PyAudio()
        stream = None
        wf = None

        try:
            fmt = pyaudio.paInt16
            self.sample_width = pa.get_sample_size(fmt)
            self.start_unix_time_s = time.time()
            self._write_info_csv()

            stream = pa.open(
                format=fmt,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.device_index,
            )

            wf = wave.open(self.wav_path, "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)

            print(f"[Audio] START -> {self.wav_path}")
            print(f"[Audio] audio_start_unix_time_s = {self.start_unix_time_s:.6f}")

            while not self.stop_event.is_set():
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    wf.writeframes(data)
                except Exception as e:
                    print("[Audio] read/write error:", e)
                    time.sleep(0.05)

        except Exception as e:
            print("[Audio] recorder error:", e)
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if wf is not None:
                    wf.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass
            print("[Audio] STOP")

    def start(self):
        import threading
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)


class ChunkedCsvLogger:
    def __init__(
        self,
        sampling_rate: int,
        out_csv: str,
        emg_channels: List[int],
        time_channel: int,
        accel_channels: List[int],
        gyro_channels: List[int],
        mag_channels: List[int],
        battery_channel: Optional[int],
        flush_every_s: float = 1.0,
        apply_filters: bool = True,
        min_filter_len: int = 128,
        tz_name: str = "America/New_York",
    ):
        self.sampling_rate = int(sampling_rate)
        self.out_csv = out_csv
        self.flush_every_s = float(flush_every_s)
        self.apply_filters = bool(apply_filters)
        self.min_filter_len = int(min_filter_len)
        self.tz_name = tz_name

        self.emg_channels = list(emg_channels)
        self.time_channel = int(time_channel)
        self.accel_channels = list(accel_channels)
        self.gyro_channels = list(gyro_channels)
        self.mag_channels = list(mag_channels)
        self.battery_channel = battery_channel

        self._last_flush_t = time.time()
        self._header_written = False
        self._pending_chunks = []
        self._last_logged_ts = None

    def add_chunk(self, new: np.ndarray):
        if new is not None and new.size > 0:
            self._pending_chunks.append(new)

    def maybe_flush(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_flush_t) < self.flush_every_s:
            return

        self._last_flush_t = now

        if not self._pending_chunks:
            return

        try:
            new = np.concatenate(self._pending_chunks, axis=1)
        except Exception:
            for chunk in self._pending_chunks:
                self._write_chunk(chunk)
            self._pending_chunks = []
            return

        self._pending_chunks = []
        self._write_chunk(new)

    def _write_chunk(self, new: np.ndarray):
        if new is None or new.size == 0:
            return

        ts = new[self.time_channel, :].astype(np.float64, copy=False).flatten()
        if ts.size == 0:
            return

        order = np.argsort(ts)
        ts = ts[order]
        new = new[:, order]

        if self._last_logged_ts is not None:
            keep = ts > self._last_logged_ts
            if not np.any(keep):
                return
            ts = ts[keep]
            new = new[:, keep]

        self._last_logged_ts = ts[-1]

        df_data = {"unix_time_s": ts}

        if ts.size >= 2:
            dt = np.diff(ts)
            df_data["dt_s"] = np.concatenate([[np.nan], dt])
        else:
            df_data["dt_s"] = np.array([np.nan] * ts.size, dtype=np.float64)

        try:
            dt_utc = pd.to_datetime(ts, unit="s", utc=True)
            dt_loc = dt_utc.tz_convert(self.tz_name)
            df_data["UTC"] = dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f").str.slice(0, 23)
            df_data["LOCAL"] = dt_loc.strftime("%Y-%m-%dT%H:%M:%S.%f").str.slice(0, 23)
        except Exception:
            df_data["UTC"] = np.array([""] * ts.shape[0], dtype=object)
            df_data["LOCAL"] = np.array([""] * ts.shape[0], dtype=object)

        emg_raw = new[self.emg_channels, :]
        emg_filt = _maybe_filter_2d(
            emg_raw,
            do_filter=self.apply_filters,
            fs=self.sampling_rate,
            min_len=self.min_filter_len,
        )

        for i in range(emg_raw.shape[0]):
            df_data[f"EMG{i+1}_Raw"] = emg_raw[i, :]
        for i in range(emg_filt.shape[0]):
            df_data[f"EMG{i+1}_Filtered"] = emg_filt[i, :]

        if len(self.accel_channels):
            accel_raw = new[self.accel_channels, :]
            for i in range(accel_raw.shape[0]):
                df_data[f"Accel{i+1}_Raw"] = accel_raw[i, :]

        if len(self.gyro_channels):
            gyro_raw = new[self.gyro_channels, :]
            for i in range(gyro_raw.shape[0]):
                df_data[f"Gyro{i+1}_Raw"] = gyro_raw[i, :]

        if len(self.mag_channels):
            mag_raw = new[self.mag_channels, :]
            for i in range(mag_raw.shape[0]):
                df_data[f"Mag{i+1}_Raw"] = mag_raw[i, :]

        if self.battery_channel is not None:
            try:
                batt = new[self.battery_channel, :].astype(np.float32, copy=False).flatten()
                df_data["Battery"] = batt
            except Exception:
                pass

        out = pd.DataFrame(df_data)
        out.to_csv(self.out_csv, mode="a", header=not self._header_written, index=False)
        self._header_written = True


class ToggleRecordingGraph(QtGui.QWidget):
    def __init__(
        self,
        board_shim: BoardShim,
        audio_recorder: ContinuousAudioRecorder,
        csv_flush_every_s: float = 1.0,
        apply_filters_in_csv: bool = True,
        apply_filters_in_plot: bool = True,
        min_filter_len: int = 128,
        tz_name: str = "America/New_York",
    ):
        super().__init__()

        self.board_shim = board_shim
        self.audio_recorder = audio_recorder
        self.board_id = board_shim.get_board_id()

        self.emg_channels = _safe_channels(BoardShim.get_emg_channels, self.board_id)
        self.time_channel = BoardShim.get_timestamp_channel(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.accel_channels = _safe_channels(BoardShim.get_accel_channels, self.board_id)
        self.gyro_channels = _safe_channels(BoardShim.get_gyro_channels, self.board_id)
        self.mag_channels = _safe_channels(BoardShim.get_magnetometer_channels, self.board_id)
        self.battery_channel = _safe_channel(BoardShim.get_battery_channel, self.board_id)

        self.apply_filters_in_csv = bool(apply_filters_in_csv)
        self.apply_filters_in_plot = bool(apply_filters_in_plot)
        self.min_filter_len = int(min_filter_len)
        self.tz_name = tz_name
        self.csv_flush_every_s = float(csv_flush_every_s)

        self.recording = False
        self.csv_logger: Optional[ChunkedCsvLogger] = None
        self.csv_path: Optional[str] = None

        self.update_speed_ms = 50
        self.visual_window_s = 10
        self.visual_points = int(self.visual_window_s * self.sampling_rate)

        self.viz_buffer = np.zeros((len(self.emg_channels), self.visual_points), dtype=np.float64)
        self.viz_time = np.full(self.visual_points, np.nan, dtype=np.float64)

        self._t0 = None
        self._last_viz_ts = None
        self._latest_battery = None

        self.setWindowTitle("EMG Recorder")
        self.resize(1500, 900)

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)

        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        self.text_label = QtGui.QLabel()
        self.text_label.setMinimumWidth(560)
        self.text_label.setWordWrap(True)
        font = self.text_label.font()
        font.setPointSize(14)
        self.text_label.setFont(font)
        layout.addWidget(self.text_label)

        self._init_timeseries()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

    def _init_timeseries(self):
        self.plots = []
        self.curves = []

        for i in range(len(self.emg_channels)):
            p = self.graph_widget.addPlot(row=i, col=0)
            p.setYRange(-150, 150)
            p.setTitle(f"EMG Channel {i+1}")
            if i < len(self.emg_channels) - 1:
                p.setXLink(self.plots[0] if self.plots else None)
            self.plots.append(p)
            self.curves.append(p.plot())

    def _clear_viz(self):
        self.viz_buffer[:] = 0.0
        self.viz_time[:] = np.nan
        self._t0 = None
        for curve in self.curves:
            curve.setData([])

    def _append_to_viz(self, t_rel: np.ndarray, emg_raw: np.ndarray):
        n_new = int(t_rel.shape[0])
        if n_new <= 0:
            return

        if n_new >= self.visual_points:
            self.viz_time = t_rel[-self.visual_points:].copy()
            self.viz_buffer = emg_raw[:, -self.visual_points:].copy()
        else:
            self.viz_time = np.roll(self.viz_time, -n_new)
            self.viz_time[-n_new:] = t_rel
            self.viz_buffer = np.roll(self.viz_buffer, -n_new, axis=1)
            self.viz_buffer[:, -n_new:] = emg_raw

    def start_recording(self):
        if self.recording:
            return
        self.csv_path = save_file_name("emg", "data", "csv")
        self.csv_logger = ChunkedCsvLogger(
            sampling_rate=self.sampling_rate,
            out_csv=self.csv_path,
            emg_channels=self.emg_channels,
            time_channel=self.time_channel,
            accel_channels=self.accel_channels,
            gyro_channels=self.gyro_channels,
            mag_channels=self.mag_channels,
            battery_channel=self.battery_channel,
            flush_every_s=self.csv_flush_every_s,
            apply_filters=self.apply_filters_in_csv,
            min_filter_len=self.min_filter_len,
            tz_name=self.tz_name,
        )
        self.recording = True
        self._clear_viz()
        self._last_viz_ts = None
        print(f"[EMG] START -> {self.csv_path}")

    def stop_recording(self):
        if not self.recording:
            return
        if self.csv_logger is not None:
            self.csv_logger.maybe_flush(force=True)
        self.recording = False
        self.csv_logger = None
        print("[EMG] STOP")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.toggle_recording()
            event.accept()
            return
        super().keyPressEvent(event)

    def update(self):
        try:
            cnt = int(self.board_shim.get_board_data_count())
        except Exception:
            cnt = 0

        if cnt > 0:
            data = self.board_shim.get_board_data(cnt)
            if data is not None and data.size > 0:
                t = data[self.time_channel].astype(np.float64, copy=False).flatten()
                if t.size > 0:
                    order = np.argsort(t)
                    t = t[order]
                    data = data[:, order]

                    if self._last_viz_ts is not None:
                        keep = t > self._last_viz_ts
                        if np.any(keep):
                            t = t[keep]
                            data = data[:, keep]
                        else:
                            t = np.array([])

                    if t.size > 0:
                        self._last_viz_ts = t[-1]

                        if self.recording:
                            if self._t0 is None:
                                self._t0 = t[0]
                            t_rel = t - self._t0
                            emg_raw = data[self.emg_channels, :]
                            self._append_to_viz(t_rel, emg_raw)

                            if self.csv_logger is not None:
                                self.csv_logger.add_chunk(data)
                                self.csv_logger.maybe_flush(force=False)

                        if self.battery_channel is not None:
                            try:
                                batt = data[self.battery_channel, :].astype(np.float64, copy=False).flatten()
                                if batt.size > 0:
                                    self._latest_battery = batt[-1]
                            except Exception:
                                pass

        if self.recording:
            emg_plot = _maybe_filter_2d(
                self.viz_buffer.copy(),
                do_filter=self.apply_filters_in_plot,
                fs=self.sampling_rate,
                min_len=self.min_filter_len,
            )

            valid = np.isfinite(self.viz_time)
            x = self.viz_time[valid]

            for i in range(len(self.emg_channels)):
                y = emg_plot[i, valid] if valid.any() else emg_plot[i]
                if x.size == y.size and x.size > 0:
                    self.curves[i].setData(x, y)
                else:
                    self.curves[i].setData([])
        else:
            for curve in self.curves:
                curve.setData([])

        status = "ON" if self.recording else "OFF"
        latest_text = "EMG Recorder\n"
        latest_text += f"\nEMG recording: {status}"
        latest_text += "\nPress SPACE to start/stop EMG recording"

        latest_text += f"\n\nSampling rate: {self.sampling_rate} Hz"
        latest_text += f"\nVisible EMG window: {self.visual_window_s}s"

        if self.csv_path is not None:
            latest_text += f"\nCurrent/last EMG CSV:\n{self.csv_path}"

        latest_text += f"\n\nContinuous audio WAV:\n{self.audio_recorder.wav_path}"
        if self.audio_recorder.start_unix_time_s is not None:
            latest_text += f"\nAudio start unix time: {self.audio_recorder.start_unix_time_s:.6f}"
        latest_text += f"\nAudio info CSV:\n{self.audio_recorder.info_csv_path}"

        if self._latest_battery is not None:
            latest_text += f"\nBattery: {self._latest_battery}"

        self.text_label.setText(latest_text)

    def closeEvent(self, event):
        try:
            self.stop_recording()
        except Exception:
            pass
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.4.1")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--audio-device-index", type=int, default=None)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-channels", type=int, default=1)
    parser.add_argument("--audio-chunk", type=int, default=1024)
    parser.add_argument("--stream_buffer_s", type=int, default=30)
    parser.add_argument("--csv_flush_every_s", type=float, default=1.0)
    parser.add_argument("--plot_filters", action="store_true")
    parser.add_argument("--no_plot_filters", dest="plot_filters", action="store_false")
    parser.set_defaults(plot_filters=True)
    parser.add_argument("--csv_filters", action="store_true")
    parser.add_argument("--no_csv_filters", dest="csv_filters", action="store_false")
    parser.set_defaults(csv_filters=True)

    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    params = MindRoveInputParams()
    params.ip_address = args.ip
    params.ip_port = args.port

    board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
    board_shim.prepare_session()

    sr_hz = BoardShim.get_sampling_rate(board_shim.get_board_id())
    print(f"Sampling rate: {sr_hz} Hz")

    board_shim.start_stream(int(sr_hz * args.stream_buffer_s))
    print(f"Started stream with internal buffer ~{args.stream_buffer_s}s")

    audio_recorder = ContinuousAudioRecorder(
        out_dir="audio",
        device_index=args.audio_device_index,
        sample_rate=args.audio_sample_rate,
        channels=args.audio_channels,
        chunk=args.audio_chunk,
    )
    audio_recorder.start()

    app = QtGui.QApplication([])
    widget = ToggleRecordingGraph(
        board_shim=board_shim,
        audio_recorder=audio_recorder,
        csv_flush_every_s=args.csv_flush_every_s,
        apply_filters_in_csv=args.csv_filters,
        apply_filters_in_plot=args.plot_filters,
        min_filter_len=128,
        tz_name="America/New_York",
    )
    widget.show()

    try:
        app.exec_()
    finally:
        try:
            widget.stop_recording()
        except Exception:
            pass
        try:
            audio_recorder.stop()
        except Exception:
            pass
        try:
            board_shim.stop_stream()
        except Exception:
            pass
        try:
            board_shim.release_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()