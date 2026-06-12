"""pyqtgraph chart widgets: candlesticks, trade arrows, equity curves.

X axes are epoch seconds (naive CT treated as UTC for display purposes, so the
axis shows the same wall-clock numbers as the data).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPicture, QPen

from . import theme


def to_epoch(ts: np.ndarray) -> np.ndarray:
    """datetime64[s] (naive CT) -> float seconds for plotting."""
    return ts.astype("datetime64[s]").astype(np.int64).astype(np.float64)


class CandlestickItem(pg.GraphicsObject):
    """Fast OHLC candles rendered into a QPicture once per data set."""

    def __init__(self, t: np.ndarray, o, h, l, c, width: float = 42.0):
        super().__init__()
        self.t, self.o, self.h, self.l, self.c = t, o, h, l, c
        self.width = width
        self.picture = QPicture()
        self._render()

    def _render(self) -> None:
        p = QPainter(self.picture)
        up_pen = QPen(QColor(theme.UP)); up_pen.setWidthF(1.0); up_pen.setCosmetic(True)
        dn_pen = QPen(QColor(theme.DOWN)); dn_pen.setWidthF(1.0); dn_pen.setCosmetic(True)
        up_brush, dn_brush = QColor(theme.UP), QColor(theme.DOWN)
        w2 = self.width / 2.0
        for t, o, h, l, c in zip(self.t, self.o, self.h, self.l, self.c):
            if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
                continue
            up = c >= o
            p.setPen(up_pen if up else dn_pen)
            p.setBrush(up_brush if up else dn_brush)
            if h > max(o, c):
                p.drawLine(QPointF(t, max(o, c)), QPointF(t, h))
            if l < min(o, c):
                p.drawLine(QPointF(t, min(o, c)), QPointF(t, l))
            body_top, body_bot = max(o, c), min(o, c)
            if body_top == body_bot:  # doji: flat tick
                p.drawLine(QPointF(t - w2, o), QPointF(t + w2, o))
            else:
                p.drawRect(pg.QtCore.QRectF(t - w2, body_bot, self.width, body_top - body_bot))
        p.end()

    def paint(self, painter, *args) -> None:
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if len(self.t) == 0:
            return pg.QtCore.QRectF()
        lo = float(np.nanmin(self.l)); hi = float(np.nanmax(self.h))
        return pg.QtCore.QRectF(float(self.t[0]) - self.width, lo,
                                float(self.t[-1] - self.t[0]) + 2 * self.width, hi - lo)


class SignalChart(pg.GraphicsLayoutWidget):
    """Candlestick chart + linked equity subplot + trade arrow markers.

    Marker conventions (per spec): blue arrows = buys, red arrows = sells,
    white arrows = exits. Entry markers carry a text label with the structure
    and fill; exit labels carry the exit reason and trade P&L.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.price_plot = self.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.price_plot.showGrid(x=True, y=True, alpha=0.15)
        self.price_plot.setLabel("left", "Price")
        self.nextRow()
        self.equity_plot = self.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.equity_plot.showGrid(x=True, y=True, alpha=0.15)
        self.equity_plot.setLabel("left", "Equity $")
        self.equity_plot.setXLink(self.price_plot)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)

        self._crosshair_v = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.FG_DIM, width=1, style=Qt.PenStyle.DashLine))
        self._crosshair_h = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.FG_DIM, width=1, style=Qt.PenStyle.DashLine))
        self.price_plot.addItem(self._crosshair_v, ignoreBounds=True)
        self.price_plot.addItem(self._crosshair_h, ignoreBounds=True)
        self._readout = pg.LabelItem(justify="left")
        self.addItem(self._readout, row=2, col=0)
        self._proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved,
                                     rateLimit=30, slot=self._mouse_moved)
        self._candles: CandlestickItem | None = None
        self._bar_data = None

    # ------------------------------------------------------------------ data
    def set_candles(self, ts: np.ndarray, o, h, l, c, bar_secs: int = 60) -> None:
        self.price_plot.clear()
        self.price_plot.addItem(self._crosshair_v, ignoreBounds=True)
        self.price_plot.addItem(self._crosshair_h, ignoreBounds=True)
        t = to_epoch(ts)
        self._candles = CandlestickItem(t, np.asarray(o, float), np.asarray(h, float),
                                        np.asarray(l, float), np.asarray(c, float),
                                        width=bar_secs * 0.7)
        self.price_plot.addItem(self._candles)
        self._bar_data = (t, np.asarray(o, float), np.asarray(h, float),
                          np.asarray(l, float), np.asarray(c, float))
        self.price_plot.autoRange()

    def set_equity(self, ts: np.ndarray, equity: np.ndarray) -> None:
        self.equity_plot.clear()
        t = to_epoch(ts)
        pen = pg.mkPen(theme.EQUITY, width=2)
        self.equity_plot.plot(t, equity, pen=pen)
        zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.FG_DIM, style=Qt.PenStyle.DotLine))
        self.equity_plot.addItem(zero)
        neg = np.where(equity < 0, equity, 0.0)
        fill = self.equity_plot.plot(t, neg, pen=pg.mkPen(None))
        zero_curve = self.equity_plot.plot(t, np.zeros_like(equity), pen=pg.mkPen(None))
        between = pg.FillBetweenItem(fill, zero_curve, brush=pg.mkBrush(239, 83, 80, 60))
        self.equity_plot.addItem(between)

    # --------------------------------------------------------------- markers
    def _bar_at(self, t_epoch: float) -> int | None:
        if self._bar_data is None or len(self._bar_data[0]) == 0:
            return None
        t = self._bar_data[0]
        i = int(np.clip(np.searchsorted(t, t_epoch), 0, len(t) - 1))
        if i > 0 and abs(t[i - 1] - t_epoch) < abs(t[i] - t_epoch):
            i -= 1
        return i

    def add_trade_markers(self, trades: list[dict]) -> None:
        """trades: dicts with entry_ts, exit_ts (datetime64), direction
        ('buy'|'sell'), label, entry_text, exit_text, pnl."""
        if self._bar_data is None:
            return
        t_arr, _o, h_arr, l_arr, _c = self._bar_data
        span = max(float(np.nanmax(h_arr) - np.nanmin(l_arr)), 1e-9)
        off = span * 0.04

        for tr in trades:
            te = to_epoch(np.array([tr["entry_ts"]], "datetime64[s]"))[0]
            tx = to_epoch(np.array([tr["exit_ts"]], "datetime64[s]"))[0]
            ie, ix = self._bar_at(te), self._bar_at(tx)
            if ie is None or ix is None:
                continue
            buy = tr["direction"] == "buy"
            color = theme.BUY if buy else theme.SELL
            # entry: buys point up from below the low; sells point down from above the high
            ey = l_arr[ie] - off if buy else h_arr[ie] + off
            entry_angle = -90 if buy else 90  # ArrowItem: -90 points up
            self._arrow(te, ey, entry_angle, color)
            self._label(te, ey, tr.get("entry_text", ""), color, anchor=(0.5, 0 if buy else 1))
            # exit: white, mirrored placement
            xy = h_arr[ix] + off if buy else l_arr[ix] - off
            exit_angle = 90 if buy else -90
            self._arrow(tx, xy, exit_angle, theme.EXIT)
            pnl = tr.get("pnl", 0.0)
            self._label(tx, xy, f"{tr.get('exit_text', '')}  {pnl:+,.0f}",
                        theme.WIN if pnl >= 0 else theme.LOSS,
                        anchor=(0.5, 1 if buy else 0))
            # dashed connector entry -> exit at entry price level
            conn = pg.PlotDataItem([te, tx], [ey, xy],
                                   pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
            self.price_plot.addItem(conn)

    def _arrow(self, x: float, y: float, angle: int, color: str) -> None:
        a = pg.ArrowItem(angle=angle, tipAngle=42, baseAngle=8, headLen=18,
                         brush=pg.mkBrush(color), pen=pg.mkPen("#000000", width=1))
        a.setPos(x, y)
        self.price_plot.addItem(a)

    def _label(self, x: float, y: float, text: str, color: str, anchor) -> None:
        if not text:
            return
        item = pg.TextItem(text=text, color=color, anchor=anchor)
        item.setPos(x, y)
        self.price_plot.addItem(item)

    # -------------------------------------------------------------- crosshair
    def _mouse_moved(self, evt) -> None:
        pos = evt[0]
        if not self.price_plot.sceneBoundingRect().contains(pos):
            return
        mp = self.price_plot.vb.mapSceneToView(pos)
        self._crosshair_v.setPos(mp.x())
        self._crosshair_h.setPos(mp.y())
        i = self._bar_at(mp.x())
        if i is None:
            return
        t, o, h, l, c = (d[i] for d in self._bar_data)
        when = np.datetime64(int(t), "s")
        self._readout.setText(
            f"<span style='color:{theme.FG_DIM}'>{when}</span>  "
            f"O <b>{o:.2f}</b>  H <b>{h:.2f}</b>  L <b>{l:.2f}</b>  C <b>{c:.2f}</b>"
            f"  <span style='color:{theme.FG_DIM}'>cursor {mp.y():.2f}</span>")


class EquityReportChart(pg.GraphicsLayoutWidget):
    """Multi-day report graphs: cumulative equity, underwater curve, daily P&L bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq = self.addPlot(row=0, col=0)
        self.eq.setLabel("left", "Cumulative $")
        self.eq.showGrid(x=True, y=True, alpha=0.15)
        self.dd = self.addPlot(row=1, col=0)
        self.dd.setLabel("left", "Drawdown $")
        self.dd.showGrid(x=True, y=True, alpha=0.15)
        self.dd.setXLink(self.eq)
        self.bars = self.addPlot(row=2, col=0)
        self.bars.setLabel("left", "Daily P&L $")
        self.bars.setLabel("bottom", "Trading day #")
        self.bars.showGrid(x=True, y=True, alpha=0.15)
        self.bars.setXLink(self.eq)
        for r, f in ((0, 3), (1, 1), (2, 1)):
            self.ci.layout.setRowStretchFactor(r, f)

    def set_results(self, daily_pnl: np.ndarray) -> None:
        for p in (self.eq, self.dd, self.bars):
            p.clear()
        if len(daily_pnl) == 0:
            return
        x = np.arange(1, len(daily_pnl) + 1, dtype=float)
        cum = np.cumsum(daily_pnl)
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        under = cum - peak

        self.eq.plot(np.concatenate([[0.0], x]), np.concatenate([[0.0], cum]),
                     pen=pg.mkPen(theme.EQUITY, width=2),
                     symbol="o", symbolSize=4, symbolBrush=theme.EQUITY, symbolPen=None)
        self.eq.addItem(pg.InfiniteLine(pos=0, angle=0,
                        pen=pg.mkPen(theme.FG_DIM, style=Qt.PenStyle.DotLine)))
        self.dd.plot(x, under, pen=pg.mkPen(theme.DRAWDOWN, width=2), fillLevel=0,
                     brush=pg.mkBrush(239, 83, 80, 70))
        heights = daily_pnl
        brushes = [pg.mkBrush(theme.WIN if v >= 0 else theme.LOSS) for v in heights]
        bar = pg.BarGraphItem(x=x, height=heights, width=0.7, brushes=brushes, pen=pg.mkPen(None))
        self.bars.addItem(bar)
        self.eq.autoRange()
