import os
import sys
import time
import ctypes
import threading
import tkinter as tk
import cv2
import mss
import numpy as np
import chess
import chess.engine

PROFILE_FILE = "board_profile.npz"

START_LAYOUT = [
    list("rnbqkbnr"),
    list("pppppppp"),
    list("........"),
    list("........"),
    list("........"),
    list("........"),
    list("PPPPPPPP"),
    list("RNBQKBNR"),
]


def resource_path(relative_path: str) -> str:
    """Resolve bundled file paths inside a PyInstaller .exe or local folder."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


class ChessAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stockfish Vision")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.playing_as = chess.WHITE
        self.board_rect = None
        self.templates = self.load_templates()
        self.running = True
        self.last_fen = ""

        # Start Stockfish silently (no console popup inside --noconsole EXE)
        sf_path = resource_path("stockfish.exe")
        popen_kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        self.engine = chess.engine.SimpleEngine.popen_uci(sf_path, **popen_kwargs)

        self._build_control_bar()
        self._build_overlay()

        # Start background screen-reading thread
        self.worker = threading.Thread(target=self._vision_loop, daemon=True)
        self.worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_control_bar(self):
        frame = tk.Frame(self.root, bg="#1e1e2e", padx=10, pady=8)
        frame.pack()

        btn_style = {
            "bg": "#313244",
            "fg": "#cdd6f4",
            "activebackground": "#45475a",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "padx": 10,
            "pady": 4,
            "font": ("Segoe UI", 9, "bold"),
        }

        tk.Button(frame, text="1. Select Board", command=self.select_board, **btn_style).pack(side=tk.LEFT, padx=3)
        tk.Button(frame, text="2. Calibrate Start", command=self.calibrate_board, **btn_style).pack(side=tk.LEFT, padx=3)

        self.side_btn = tk.Button(frame, text="Side: White", command=self.toggle_side, **btn_style)
        self.side_btn.pack(side=tk.LEFT, padx=3)

        self.status_var = tk.StringVar(
            value="Templates loaded. Select board." if self.templates else "Select board & calibrate on move 1."
        )
        tk.Label(
            frame,
            textvariable=self.status_var,
            bg="#1e1e2e",
            fg="#a6e3a1",
            font=("Consolas", 10, "bold"),
            width=34,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(10, 0))

    def _build_overlay(self):
        """Transparent, click-through overlay window aligned with the chessboard."""
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.transparent_key = "#010101"
        self.overlay.configure(bg=self.transparent_key)
        self.overlay.attributes("-transparentcolor", self.transparent_key)
        self.overlay.withdraw()

        self.canvas = tk.Canvas(
            self.overlay,
            bg=self.transparent_key,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Make the overlay click-through on Windows
        if os.name == "nt":
            self.overlay.update_idletasks()
            hwnd = self.overlay.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    def toggle_side(self):
        self.playing_as = chess.BLACK if self.playing_as == chess.WHITE else chess.WHITE
        label = "White" if self.playing_as == chess.WHITE else "Black"
        self.side_btn.config(text=f"Side: {label}")
        self.last_fen = ""  # Force re-evaluation

    def select_board(self):
        self.overlay.withdraw()
        self.root.withdraw()
        time.sleep(0.2)

        with mss.mss() as sct:
            mon = sct.monitors[1]
            raw = np.array(sct.grab(mon))
            screen = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        # Scale down preview if monitor is >1600px wide so ROI window fits cleanly
        h, w = screen.shape[:2]
        scale = min(1.0, 1600.0 / w)
        preview = cv2.resize(screen, (int(w * scale), int(h * scale))) if scale < 1.0 else screen

        roi = cv2.selectROI("Drag over the 8x8 board and press ENTER", preview, False)
        cv2.destroyAllWindows()
        self.root.deiconify()

        if roi[2] > 20 and roi[3] > 20:
            self.board_rect = {
                "left": int(roi[0] / scale) + mon["left"],
                "top": int(roi[1] / scale) + mon["top"],
                "width": int(roi[2] / scale),
                "height": int(roi[3] / scale),
            }
            self.overlay.geometry(
                f"{self.board_rect['width']}x{self.board_rect['height']}"
                f"+{self.board_rect['left']}+{self.board_rect['top']}"
            )
            self.overlay.deiconify()
            self.last_fen = ""
            self.status_var.set("Board locked. Ready.")

    def _grab_squares(self, sct):
        raw = np.array(sct.grab(self.board_rect))
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        h, w, _ = img.shape
        sq_h, sq_w = h / 8.0, w / 8.0
        squares = []
        for r in range(8):
            row = []
            for c in range(8):
                y1, y2 = int((r + 0.18) * sq_h), int((r + 0.82) * sq_h)
                x1, x2 = int((c + 0.18) * sq_w), int((c + 0.82) * sq_w)
                crop = cv2.resize(img[y1:y2, x1:x2], (48, 48))
                row.append(crop)
            squares.append(row)
        return squares

    def calibrate_board(self):
        if not self.board_rect:
            self.status_var.set("Select the board region first!")
            return

        with mss.mss() as sct:
            squares = self._grab_squares(sct)

        is_white = self.playing_as == chess.WHITE
        layout = START_LAYOUT if is_white else [row[::-1] for row in START_LAYOUT[::-1]]
        templates = {piece: [] for piece in "PNBRQKpnbrqk."}

        for r in range(8):
            for c in range(8):
                templates[layout[r][c]].append(squares[r][c])

        # Keep up to 4 samples per piece type and save to disk
        self.templates = {k: np.array(v[:4]) for k, v in templates.items()}
        np.savez_compressed(PROFILE_FILE, **self.templates)
        self.last_fen = ""
        self.status_var.set("Calibrated & saved! Watching...")

    def load_templates(self):
        if os.path.exists(PROFILE_FILE):
            data = np.load(PROFILE_FILE)
            return {k: data[k] for k in data.files}
        return None

    def _classify_square(self, crop):
        best_piece, best_score = ".", -1.0
        for piece, samples in self.templates.items():
            for sample in samples:
                score = cv2.matchTemplate(crop, sample, cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_piece = piece
        return best_piece

    def _grid_to_fen(self, grid):
        is_white = self.playing_as == chess.WHITE
        rows = grid if is_white else [row[::-1] for row in grid[::-1]]
        fen_rows = []
        for row in rows:
            empty, fen_row = 0, ""
            for cell in row:
                if cell == ".":
                    empty += 1
                else:
                    if empty:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell
            if empty:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        side = "w" if is_white else "b"
        return f"{'/'.join(fen_rows)} {side} - - 0 1"

    def _draw_arrow(self, move: chess.Move):
        self.canvas.delete("all")
        if not move or not self.board_rect:
            return

        w = self.board_rect["width"]
        h = self.board_rect["height"]
        sq_w, sq_h = w / 8.0, h / 8.0
        is_white = self.playing_as == chess.WHITE

        def sq_center(sq):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            col = f if is_white else 7 - f
            row = (7 - r) if is_white else r
            return (col + 0.5) * sq_w, (row + 0.5) * sq_h

        x1, y1 = sq_center(move.from_square)
        x2, y2 = sq_center(move.to_square)

        self.canvas.create_line(
            x1, y1, x2, y2,
            arrow=tk.LAST,
            arrowshape=(18, 22, 8),
            width=7,
            fill="#00FF88",
        )

    def _vision_loop(self):
        with mss.mss() as sct:
            while self.running:
                if not self.board_rect or not self.templates:
                    time.sleep(0.2)
                    continue

                squares = self._grab_squares(sct)
                grid = [[self._classify_square(squares[r][c]) for c in range(8)] for r in range(8)]
                fen = self._grid_to_fen(grid)

                # Guard against mid-drag animations or popups (must have 1 king each)
                board_part = fen.split()[0]
                if board_part.count("K") != 1 or board_part.count("k") != 1:
                    time.sleep(0.15)
                    continue

                if fen != self.last_fen:
                    try:
                        board = chess.Board(fen)
                        if board.is_valid():
                            self.last_fen = fen
                            info = self.engine.analyse(board, chess.engine.Limit(time=0.25))
                            best_move = info["pv"][0]
                            score = info["score"].white()
                            san = board.san(best_move)

                            self.root.after(0, self._draw_arrow, best_move)
                            self.root.after(
                                0,
                                self.status_var.set,
                                f"Best: {san} ({best_move}) | {score}",
                            )
                    except (ValueError, chess.engine.EngineError):
                        pass

                time.sleep(0.18)

    def on_close(self):
        self.running = False
        self.engine.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChessAssistantApp(root)
    root.mainloop()
