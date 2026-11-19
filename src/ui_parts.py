"""Small HTML/SVG building blocks for the new app (app_new.py).

Everything here only makes text (HTML or SVG); there is no machine learning in this file.
The old app (app.py) does not use this file.
"""
import html

INK, MUTE, LINE, BG = "#f5f5f5", "#8a8a8a", "#2a2b30", "#111216"


def esc(x):
    """Make a value safe to put inside HTML (player names can contain < > &)."""
    return html.escape(str(x))


# ---------- logo: a queen and a king as white line art ----------
_BASE = ("M-20 95H20V90H-20Z "
         "M-17 90C-17 86 -14 83.5 -11 82.5H11C14 83.5 17 86 17 90 "
         "M-10 82.5C-6 70 -5 57 -7 47 M10 82.5C6 70 5 57 7 47 "
         "M-11 47H11L9 42.5H-9Z ")
KING = _BASE + "M-8 42.5C-15 36 -14 27 -6 24H6C14 27 15 36 8 42.5 M0 24V8 M-5.5 13.5H5.5"
QUEEN = (_BASE + "M-8 42.5C-13 37 -13 31 -11 22L-8 29L-5.5 20L-3 27L0 17L3 27L5.5 20L8 29L11 22"
         "C13 31 13 37 8 42.5")


def logo_svg(height=40, stroke=INK, width_px=1.3):
    """Queen (left, a little smaller) and king (right), outline only."""
    def piece(d, cx, s, ball=False):
        extra = '<circle cx="0" cy="13.5" r="3"/>' if ball else ""
        return (f'<g transform="translate({cx} 95) scale({s}) translate(0 -95)">'
                f'<path d="{d}"/>{extra}</g>')
    return (f'<svg class="logo" viewBox="14 4 92 94" height="{height}" role="img" '
            f'aria-label="Queen and king logo" xmlns="http://www.w3.org/2000/svg">'
            f'<g fill="none" stroke="{stroke}" stroke-width="{width_px}" stroke-linejoin="round" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke" style="vector-effect:non-scaling-stroke">'
            f'{piece(QUEEN, 40, 0.9, ball=True)}{piece(KING, 78, 1.0)}</g></svg>')


# more pieces in the same line-art style (same base as the king and queen above)
PAWN = ("M-16 95H16V90H-16Z M-13 90C-13 86 -10 84 -8 83H8C10 84 13 86 13 90 "
        "M-7 83C-4 74 -4 66 -5 60 M7 83C4 74 4 66 5 60 M-9 60H9L7 56.5H-7Z")
BISHOP = (_BASE + "M-7 42.5C-14 34 -10 20 0 12C10 20 14 34 7 42.5 M3 22L-4 32")
KNIGHT = ("M-20 95H20V90H-20Z M-17 90C-17 86 -14 83.5 -11 82.5H11C14 83.5 17 86 17 90 "
          "M-11 82.5C-11 70 -12 60 -8 50C-6 44 -10 40 -15 37C-19 35 -19 29 -15 26L-3 17C1 12 5 9 7 5"
          "L9 12C16 17 18 28 16 40C14 54 12 68 11 82.5")
_EXTRA = {"pawn": '<circle cx="0" cy="47" r="9.5"/>', "bishop": '<circle cx="0" cy="8.5" r="3"/>',
          "knight": '<circle cx="-3" cy="24" r="1.4"/>', "king": "", "queen": '<circle cx="0" cy="13.5" r="3"/>'}
_PATHS = {"pawn": PAWN, "bishop": BISHOP, "knight": KNIGHT, "king": KING, "queen": QUEEN}


def piece_icon(kind, height=64, stroke=INK, width_px=1.3):
    """One white line-art piece as an <img> (st.html keeps images but removes inline <svg>)."""
    import base64
    svg = (f'<svg viewBox="-30 2 60 96" height="{height}" xmlns="http://www.w3.org/2000/svg">'
           f'<g fill="none" stroke="{stroke}" stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round">'
           f'<path d="{_PATHS[kind]}"/>{_EXTRA[kind]}</g></svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{kind} icon" height="{height}">'


def logo_img(height=34):
    """The same logo as an <img> (st.html removes inline <svg>, but keeps images)."""
    import base64
    data = base64.b64encode(logo_svg(height).encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" alt="Queen and king logo" height="{height}">'


def section_title(number, title, sub=None):
    s = f'<p class="sec-sub">{esc(sub)}</p>' if sub else ""
    return (f'<div class="sec"><span class="sec-no">{esc(number)}</span>'
            f'<h2 class="sec-title">{esc(title)}</h2><span class="sec-line"></span></div>{s}')


def note_box(text):
    """A quiet message box (instead of Streamlit's coloured info/warning boxes)."""
    return f'<div class="note-box">{text}</div>'


# ---------- charts (plain SVG, numbers come from results/*.csv) ----------
def pct(v):
    """0.8795 -> '88.0%' (adds a tiny amount so .xx5 rounds up, like in the README tables)."""
    return f"{round(v * 100 + 1e-9, 1):.1f}%"


_pct = pct


def chart_cutoff(rows):
    """rows: list of (cutoff_move, model_acc, always_white_acc). Two lines, one y-axis."""
    W, H, L, R, T, B = 640, 360, 60, 200, 26, 58
    xs = [r[0] for r in rows]
    x = lambda m: L + (m - min(xs)) / (max(xs) - min(xs)) * (W - L - R)
    y = lambda v: T + (0.80 - v) / (0.80 - 0.40) * (H - T - B)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Accuracy by cutoff move, model versus always guessing White">']
    for g in [0.4, 0.5, 0.6, 0.7, 0.8]:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{LINE}" stroke-width="1"/>'
                   f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" class="tick">{int(g * 100)}%</text>')
    for m in xs:
        out.append(f'<text x="{x(m):.1f}" y="{H - B + 24}" text-anchor="middle" class="tick">{m}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 4}" text-anchor="middle" class="axis">MOVES PER SIDE SEEN BY THE MODEL</text>')
    for idx, stroke, dash, name in [(2, MUTE, ' stroke-dasharray="5 5"', "Always White"), (1, INK, "", "Model")]:
        pts = " ".join(f"{x(r[0]):.1f},{y(r[idx]):.1f}" for r in rows)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2"{dash}/>')
        for r in rows:
            out.append(f'<g class="pt"><circle cx="{x(r[0]):.1f}" cy="{y(r[idx]):.1f}" r="4.5" fill="{BG}" '
                       f'stroke="{stroke}" stroke-width="2"/><circle cx="{x(r[0]):.1f}" cy="{y(r[idx]):.1f}" r="13" '
                       f'fill="transparent"><title>{name}, move {r[0]}: {_pct(r[idx])} (train CV)</title></circle></g>')
        last = rows[-1]
        out.append(f'<text x="{x(last[0]) + 12:.1f}" y="{y(last[idx]) + 4:.1f}" class="dlabel">'
                   f'{name} {_pct(last[idx])}</text>')
    first = rows[0]
    out.append(f'<text x="{x(first[0]) + 10:.1f}" y="{y(first[1]) - 14:.1f}" text-anchor="start" class="dlabel">'
               f'{_pct(first[1])}</text>')
    out.append("</svg>")
    return "".join(out)


def chart_confidence(rows):
    """rows: list of (label, share_of_test, model_acc, always_white_acc). Grouped bars, 0-100%."""
    W, H, L, R, T, B = 640, 370, 60, 10, 30, 70
    n = len(rows)
    band = (W - L - R) / n
    bw = min(46, band * 0.28)
    y = lambda v: T + (1 - v) * (H - T - B)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Test accuracy when the model is at least this sure, versus always guessing White">']
    for g in [0, 0.25, 0.5, 0.75, 1.0]:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{LINE}" stroke-width="1"/>'
                   f'<text x="{L - 8}" y="{y(g) + 4:.1f}" text-anchor="end" class="tick">{int(g * 100)}%</text>')

    def bar(x0, v, fill, tip):
        top, base = y(v), y(0)
        r = 4
        d = (f"M{x0:.1f},{base:.1f} V{top + r:.1f} Q{x0:.1f},{top:.1f} {x0 + r:.1f},{top:.1f} "
             f"H{x0 + bw - r:.1f} Q{x0 + bw:.1f},{top:.1f} {x0 + bw:.1f},{top + r:.1f} V{base:.1f} Z")
        return f'<path d="{d}" fill="{fill}"><title>{tip}</title></path>'

    for i, (label, share, acc, white) in enumerate(rows):
        cx = L + band * (i + 0.5)
        out.append(bar(cx - bw - 1, acc, INK, f"{label}: model {_pct(acc)} (test, {share:.0%} of games)"))
        out.append(bar(cx + 1, white, MUTE, f"{label}: always White {_pct(white)} (same games)"))
        out.append(f'<text x="{cx - bw / 2 - 1:.1f}" y="{y(acc) - 8:.1f}" text-anchor="middle" class="dlabel">{_pct(acc)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - B + 24}" text-anchor="middle" class="tick">{esc(label)}</text>'
                   f'<text x="{cx:.1f}" y="{H - B + 46}" text-anchor="middle" class="tick dim">{share:.0%} of games</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    """items: list of (label, css for the swatch)."""
    return '<div class="legend">' + "".join(
        f'<span><i style="{css}"></i>{esc(lab)}</span>' for lab, css in items) + "</div>"
