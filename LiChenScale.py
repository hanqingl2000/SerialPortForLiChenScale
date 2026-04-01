import re
import csv
import time
import threading
import queue as Q
from datetime import datetime
from collections import deque

import serial
from serial.tools import list_ports

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

# ===================== 用户可调参数 =====================
BAUD        = 9600
HZ          = 5
DT_SEC      = 1.0 / HZ
PLOT_WINDOW = 120
CSV_PREFIX  = "LiChen"
# =======================================================

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

# ── Left-panel layout constants ──────────────────────────────────────────────
_TOG_X  = 0.015;  _TOG_W = 0.038
_ALI_X  = 0.063;  _ALI_W = 0.200
_ROW_H  = 0.055
_ROW_G  = 0.016   # gap between rows (room for ax.set_title above each box)
_TOP_Y  = 0.855   # bottom-y of first row
_DRAG_THRESH = 0.018   # min figure-y movement to count as drag vs click


def scan_ports():
    ports = list(list_ports.comports())

    def score(p):
        t = " ".join(filter(None,
            [p.device, p.description, p.manufacturer or "", p.hwid])).lower()
        s = 0
        if "prolific"  in t: s += 1000
        if "radwag"    in t: s += 500
        if "usbserial" in t or "usb-serial" in t or "usb serial" in t: s += 200
        if "ftdi"      in t: s += 100
        if "ch340"     in t or "wch" in t: s += 80
        if p.device.upper().startswith("COM"): s += 10
        return s

    return sorted(ports, key=score, reverse=True)


def parse_weight(line: str):
    if not line:
        return None
    m = _num_re.search(line.strip())
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# ──────────────────────────── Background reader thread ────────────────────────

class DeviceReader(threading.Thread):
    def __init__(self, port: str, shared_q: Q.Queue):
        super().__init__(daemon=True)
        self.port     = port
        self._q       = shared_q
        self._running = False
        self._ser     = None

    def run(self):
        self._running = True
        try:
            self._ser = serial.Serial(self.port, BAUD, timeout=DT_SEC * 0.9)
        except Exception as e:
            self._q.put(("error", self.port, str(e)))
            return
        while self._running:
            try:
                self._ser.reset_input_buffer()
                raw = self._ser.readline()
                if raw:
                    w = parse_weight(raw.decode("ascii", errors="ignore"))
                    if w is not None:
                        self._q.put(("data", self.port, w, time.time()))
            except Exception as e:
                if self._running:
                    self._q.put(("error", self.port, str(e)))
                break

    def stop(self):
        self._running = False
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass


# ────────────────────────────────── App ───────────────────────────────────────

class ScaleApp:
    def __init__(self):
        self.running    = False
        self.t0         = None
        self._color_idx = 0

        self._q           = Q.Queue()
        self._readers     = {}
        self._tbufs       = {}
        self._wbufs       = {}
        self._lines       = {}
        self._colors      = {}
        self._csv_file    = None
        self._csv_writer  = None
        self._csv_fname   = ""
        self._alias_map   = {}

        self._port_objs      = []
        self._selected_ports = set()
        self._port_names     = {}   # port -> alias string (persists across refresh)
        self._port_rows      = []   # rebuilt by _rebuild_port_rows

        # ── drag state ──────────────────────────────────────────────────────
        self._drag_port      = None   # port being dragged
        self._drag_orig_idx  = None   # original row index
        self._drag_start_fy  = None   # figure-y at press
        self._drag_indicator = None   # Line2D drop target line

        self._ani = None
        self._build_ui()

    # ------------------------------------------------------------------- UI --

    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#f5f5f5")

        self.fig.text(0.015, 0.958, "Serial Ports",
                      fontsize=11, fontweight="bold")
        self.fig.text(0.015, 0.925,
                      "Click toggle = select  |  Drag toggle = reorder  |  Text = alias",
                      fontsize=7, color="#666")

        # ── Refresh button ──
        ax_ref = self.fig.add_axes([0.015, 0.03, 0.250, 0.065])
        self.btn_refresh = Button(ax_ref, "⟳  Refresh Ports",
                                  color="#607D8B", hovercolor="#78909C")
        self.btn_refresh.label.set_color("white")
        self.btn_refresh.label.set_fontsize(10)
        self.btn_refresh.on_clicked(self._on_refresh)

        # ── Main plot ──
        self.ax = self.fig.add_axes([0.33, 0.22, 0.64, 0.70])
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Weight (g)")
        self.ax.set_title("Real-time Weight Logger", fontsize=12)
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_xlim(0, PLOT_WINDOW)
        self.ax.set_ylim(0, 1)

        self.status_text = self.ax.text(
            0.01, 1.03, "Status: Idle",
            transform=self.ax.transAxes, fontsize=9, color="gray")

        # ── Filename row ──
        self.fig.text(0.33, 0.168, "Filename:",
                      fontsize=9, va="center", color="#333")
        ax_fname = self.fig.add_axes([0.425, 0.138, 0.535, 0.058])
        self._fname_box = TextBox(ax_fname, "",
                                  initial=self._default_fname(),
                                  color="#ffffff", hovercolor="#f0f0ff")

        # ── Action buttons ──
        ax_start = self.fig.add_axes([0.33, 0.03, 0.17, 0.085])
        self.btn_start = Button(ax_start, "▶  Start",
                                color="#4CAF50", hovercolor="#66BB6A")
        self.btn_start.label.set_color("white")
        self.btn_start.label.set_fontsize(11)
        self.btn_start.on_clicked(self._on_start)

        ax_stop = self.fig.add_axes([0.52, 0.03, 0.20, 0.085])
        self.btn_stop = Button(ax_stop, "■  Stop & Save",
                               color="#f44336", hovercolor="#EF5350")
        self.btn_stop.label.set_color("white")
        self.btn_stop.label.set_fontsize(11)
        self.btn_stop.on_clicked(self._on_stop)

        ax_clear = self.fig.add_axes([0.78, 0.03, 0.17, 0.085])
        self.btn_clear = Button(ax_clear, "✕  Clear & Stop",
                                color="#FF9800", hovercolor="#FFA726")
        self.btn_clear.label.set_color("white")
        self.btn_clear.label.set_fontsize(10)
        self.btn_clear.on_clicked(self._on_clear_stop)

        self.fig.canvas.mpl_connect("close_event",        self._on_close)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event",self._on_mouse_motion)
        self.fig.canvas.mpl_connect("button_release_event",self._on_mouse_release)

        self._refresh_ports()

    @staticmethod
    def _default_fname():
        return f"{CSV_PREFIX}_{datetime.now().strftime('%Y-%m-%d')}"

    # ─────────────────────────── Port panel: scan ─────────────────────────────

    def _refresh_ports(self):
        """Scan hardware, update port list, then rebuild UI rows."""
        self._port_objs = scan_ports()

        print("Available ports:")
        for p in self._port_objs:
            print(f"  {p.device:>16} | {p.description}")
        print()

        self._selected_ports &= {p.device for p in self._port_objs}
        self._rebuild_port_rows()

    # ─────────────────────────── Port panel: build ────────────────────────────

    def _rebuild_port_rows(self):
        """Tear down and recreate left-panel rows from current self._port_objs."""
        for row in self._port_rows:
            if row.get("toggle_ax"):
                row["toggle_ax"].remove()
            if row.get("no_port_text"):
                row["no_port_text"].remove()
            if row.get("alias_ax"):
                row["alias_ax"].remove()
        self._port_rows.clear()

        if not self._port_objs:
            t = self.fig.text(
                0.015, 0.50,
                "No serial ports found.\nConnect a device and\nclick Refresh.",
                fontsize=9, color="red", va="center")
            self._port_rows.append({"no_port_text": t})
            self.fig.canvas.draw_idle()
            return

        for p in self._port_objs:
            if p.device not in self._colors:
                self._colors[p.device] = _COLORS[self._color_idx % len(_COLORS)]
                self._color_idx += 1

        for i, p in enumerate(self._port_objs):
            y = _TOP_Y - i * (_ROW_H + _ROW_G)
            if y < 0.115:
                break

            port     = p.device
            color    = self._colors[port]
            selected = port in self._selected_ports
            short    = port.split("/")[-1]
            short    = short[-18:] if len(short) > 18 else short

            # toggle button — NO on_clicked; handled by _on_mouse_release
            btn_color = color if selected else "#cccccc"
            ax_t = self.fig.add_axes([_TOG_X, y, _TOG_W, _ROW_H])
            btn  = Button(ax_t, "☑" if selected else "☐",
                          color=btn_color, hovercolor=btn_color)
            btn.label.set_fontsize(13)
            btn.label.set_color("white" if selected else "#666")

            # alias TextBox with port name as title
            ax_a = self.fig.add_axes([_ALI_X, y, _ALI_W, _ROW_H])
            ax_a.set_title(short, fontsize=7.5, color=color, pad=2, loc="left")

            initial = self._port_names.get(port, "")
            box = TextBox(ax_a, "", initial=initial,
                          color="#ffffff", hovercolor="#f0f8ff")
            if not initial:
                box.ax.text(0.03, 0.5, "enter alias here",
                            transform=box.ax.transAxes,
                            fontsize=7, color="#bbb", va="center",
                            ha="left", zorder=0)
            box.on_submit(lambda txt, prt=port: self._set_alias(prt, txt))

            self._port_rows.append({
                "port":       port,
                "toggle_ax":  ax_t,
                "toggle_btn": btn,
                "alias_ax":   ax_a,
                "alias_box":  box,
            })

        self.fig.canvas.draw_idle()

    # ──────────────────────── Drag-to-reorder helpers ─────────────────────────

    def _row_bottom_y(self, i: int) -> float:
        """Figure-coordinate y of the bottom edge of row i."""
        return _TOP_Y - i * (_ROW_H + _ROW_G)

    def _find_insert_idx(self, fy: float) -> int:
        """Return the insertion index (0..n) for a drop at figure-y fy."""
        n = sum(1 for r in self._port_rows if r.get("port"))
        for i in range(n):
            if fy > self._row_bottom_y(i) + _ROW_H / 2:
                return i
        return n

    def _set_drag_indicator(self, target_idx: int):
        """Draw (or redraw) the horizontal drop-target line."""
        if self._drag_indicator is not None:
            try:
                self._drag_indicator.remove()
            except Exception:
                pass
            self._drag_indicator = None

        n = sum(1 for r in self._port_rows if r.get("port"))
        if n == 0:
            return

        if target_idx < n:
            # line sits in the gap above row target_idx
            line_y = self._row_bottom_y(target_idx) + _ROW_H + _ROW_G / 2
        else:
            # line sits below the last row
            line_y = self._row_bottom_y(n - 1) - _ROW_G / 2

        self._drag_indicator = self.fig.add_artist(
            mlines.Line2D(
                [_TOG_X, _ALI_X + _ALI_W], [line_y, line_y],
                transform=self.fig.transFigure,
                color="#333333", lw=2.5, zorder=100,
                solid_capstyle="round",
            )
        )

    # ──────────────────────── Mouse event handlers ────────────────────────────

    def _fig_coords(self, event):
        """Convert a mouse event to figure (0-1) coordinates."""
        return self.fig.transFigure.inverted().transform((event.x, event.y))

    def _on_mouse_press(self, event):
        if event.button != 1 or event.x is None:
            return
        fx, fy = self._fig_coords(event)
        # only react to clicks in the toggle-button column
        if not (_TOG_X <= fx <= _TOG_X + _TOG_W):
            return
        for i, row in enumerate(self._port_rows):
            if not row.get("port"):
                continue
            y = self._row_bottom_y(i)
            if y <= fy <= y + _ROW_H:
                self._drag_port     = row["port"]
                self._drag_orig_idx = i
                self._drag_start_fy = fy
                break

    def _on_mouse_motion(self, event):
        if self._drag_port is None or event.x is None:
            return
        _, fy = self._fig_coords(event)
        if abs(fy - self._drag_start_fy) < _DRAG_THRESH:
            return  # not yet a drag — don't show indicator yet
        self._set_drag_indicator(self._find_insert_idx(fy))
        self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if self._drag_port is None:
            return

        # clean up indicator
        if self._drag_indicator is not None:
            try:
                self._drag_indicator.remove()
            except Exception:
                pass
            self._drag_indicator = None

        port = self._drag_port
        orig = self._drag_orig_idx
        self._drag_port     = None
        self._drag_orig_idx = None

        if event.x is None:
            self.fig.canvas.draw_idle()
            return

        _, fy = self._fig_coords(event)
        movement = abs(fy - self._drag_start_fy)
        self._drag_start_fy = None

        if movement < _DRAG_THRESH:
            # ── short click → toggle selection ──
            self._toggle_port(port)
        else:
            # ── drag → reorder ──
            target = self._find_insert_idx(fy)
            # inserting at orig or orig+1 is a no-op
            if target != orig and target != orig + 1:
                item = self._port_objs.pop(orig)
                if target > orig:
                    target -= 1
                self._port_objs.insert(target, item)
                self._rebuild_port_rows()
            else:
                self.fig.canvas.draw_idle()

    # ─────────────────────────────── Callbacks ────────────────────────────────

    def _on_refresh(self, _event=None):
        if self.running:
            self.status_text.set_text("Status: Stop logging before refreshing ports.")
            self.fig.canvas.draw_idle()
            return
        self._refresh_ports()

    def _toggle_port(self, port: str):
        if port in self._selected_ports:
            self._selected_ports.discard(port)
            selected = False
        else:
            self._selected_ports.add(port)
            selected = True

        color = self._colors[port]
        for row in self._port_rows:
            if row.get("port") == port:
                btn_color = color if selected else "#cccccc"
                row["toggle_btn"].color      = btn_color
                row["toggle_btn"].hovercolor = btn_color
                row["toggle_ax"].set_facecolor(btn_color)
                row["toggle_btn"].label.set_text("☑" if selected else "☐")
                row["toggle_btn"].label.set_color("white" if selected else "#666")
                break

        self.fig.canvas.draw_idle()

    def _set_alias(self, port: str, text: str):
        self._port_names[port] = text.strip()

    def _get_alias(self, port: str) -> str:
        name = self._port_names.get(port, "").strip()
        return name if name else port.split("/")[-1]

    def _on_start(self, _=None):
        if self.running:
            return

        # preserve left-panel order for legend
        ports = [p.device for p in self._port_objs
                 if p.device in self._selected_ports]
        if not ports:
            self.status_text.set_text("Status: Select at least one port!")
            self.fig.canvas.draw_idle()
            return

        self.t0      = time.time()
        self.running = True
        base_name    = self._fname_box.text.strip() or self._default_fname()

        self._csv_fname  = f"{base_name}.csv"
        fobj             = open(self._csv_fname, "w", newline="", encoding="utf-8")
        self._csv_file   = fobj
        self._csv_writer = csv.writer(fobj)
        self._csv_writer.writerow(["Date", "Time", "Elapsed_s", "Channel", "Weight_g"])
        print(f"Logging all channels → {self._csv_fname}")

        for port in ports:
            color  = self._colors[port]
            alias  = self._get_alias(port)
            maxlen = PLOT_WINDOW * HZ * 2

            self._alias_map[port] = alias
            self._tbufs[port]     = deque(maxlen=maxlen)
            self._wbufs[port]     = deque(maxlen=maxlen)

            line, = self.ax.plot([], [], color=color, linewidth=1.4, label=alias)
            self._lines[port] = line

            reader = DeviceReader(port, self._q)
            self._readers[port] = reader
            reader.start()
            print(f"  {port}  →  channel '{alias}'")

        self.ax.legend(loc="upper right", fontsize=8)
        self.status_text.set_text(f"Status: Logging {len(ports)} device(s)…")

        self._ani = FuncAnimation(
            self.fig, self._update,
            interval=int(1000 / HZ),
            blit=False, cache_frame_data=False
        )
        self.fig.canvas.draw_idle()

    def _on_stop(self, _=None):
        if not self.running:
            return
        self.running = False

        if self._ani:
            self._ani.event_source.stop()
            self._ani = None

        for reader in self._readers.values():
            reader.stop()
        self._readers.clear()

        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            print(f"Saved: {self._csv_fname}")
        self._csv_file   = None
        self._csv_writer = None
        self._alias_map.clear()

        self.status_text.set_text(f"Status: Saved → {self._csv_fname}")
        self.fig.canvas.draw_idle()

    def _on_clear_stop(self, _=None):
        self.running = False

        if self._ani:
            self._ani.event_source.stop()
            self._ani = None

        for reader in self._readers.values():
            reader.stop()
        self._readers.clear()

        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file   = None
        self._csv_writer = None
        self._alias_map.clear()

        for line in self._lines.values():
            line.remove()
        self._lines.clear()
        self._tbufs.clear()
        self._wbufs.clear()

        leg = self.ax.get_legend()
        if leg:
            leg.remove()

        self.ax.set_xlim(0, PLOT_WINDOW)
        self.ax.set_ylim(0, 1)
        self.status_text.set_text("Status: Cleared")
        self.fig.canvas.draw_idle()

    # ─────────────────────────────── Update loop ──────────────────────────────

    def _update(self, _frame):
        if not self.running:
            return list(self._lines.values())

        while True:
            try:
                item = self._q.get_nowait()
            except Q.Empty:
                break

            if item[0] == "data":
                _, port, w, ts = item
                if port in self._tbufs:
                    elapsed = ts - self.t0
                    self._tbufs[port].append(elapsed)
                    self._wbufs[port].append(w)
                    if self._csv_writer:
                        dt_now = datetime.fromtimestamp(ts)
                        self._csv_writer.writerow([
                            dt_now.strftime("%Y-%m-%d"),
                            dt_now.strftime("%H:%M:%S.%f")[:-3],
                            f"{elapsed:.3f}",
                            self._alias_map.get(port, port.split("/")[-1]),
                            f"{w:.10g}",
                        ])
            elif item[0] == "error":
                _, port, msg = item
                self.status_text.set_text(f"Status: Error on {port}: {msg}")

        all_t, all_y = [], []
        for port, tb in self._tbufs.items():
            wb = self._wbufs[port]
            if not tb:
                continue
            t_last = tb[-1]
            tmin   = max(0.0, t_last - PLOT_WINDOW)
            xs = [t for t in tb if t >= tmin]
            ys = [v for t, v in zip(tb, wb) if t >= tmin]
            if port in self._lines:
                self._lines[port].set_data(xs, ys)
            all_t.extend(xs)
            all_y.extend(ys)

        if all_t:
            t_max = max(all_t)
            self.ax.set_xlim(max(0.0, t_max - PLOT_WINDOW), max(PLOT_WINDOW, t_max))
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            pad = 0.5 if ymin == ymax else 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - pad, ymax + pad)

        return list(self._lines.values())

    # ─────────────────────────────── Close ────────────────────────────────────

    def _on_close(self, _event=None):
        if self.running:
            self._on_stop()


def main():
    app = ScaleApp()
    plt.show()


if __name__ == "__main__":
    main()
