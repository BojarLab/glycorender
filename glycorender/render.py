import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
import xml.etree.ElementTree as ET
import re
import math
from pathlib import Path
import glycorender.pdfwrite as canvas
import glycorender.svgin as svgin
from glycorender.pdfwrite import pdfmetrics, TTFont
mm = 72.0 / 25.4
from io import BytesIO
from typing import Union

reportlab_colors = {
    'darkblue': (0, 0, 0.545),
    'green': (0, 0.5, 0),
    'darkgoldenrod': (0.722, 0.525, 0.043),
    'skyblue': (0.529, 0.808, 0.922),
    'orchid': (0.855, 0.439, 0.839),
    'purple': (0.5, 0, 0.5),
    'saddlebrown': (0.545, 0.271, 0.075),
    'orangered': (1, 0.271, 0),
    'firebrick': (0.698, 0.133, 0.133),
    'white': (1, 1, 1),
    'charcoal': (0.110, 0.098, 0.090)
}


def parse_color(color_str):
    """Parse SVG color to RGB tuple."""
    if not color_str or color_str == "none":
        return None
    if color_str.startswith('#'):
        digits = color_str[1:]
        if len(digits) == 3:
            digits = ''.join(ch * 2 for ch in digits)
        if len(digits) == 6:
            if digits.lower() == '000000':  # nothing in a GlycoDraw figure is ever pure black
                return reportlab_colors['charcoal']
            return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    if color_str.startswith('url(#'):
        # Return the gradient ID
        return color_str[5:-1]  # Remove 'url(#' and ')'
    if color_str in reportlab_colors:
        return reportlab_colors[color_str]
    return reportlab_colors['charcoal']


_PATH_TOKEN = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)')


def parse_path(d_str):
    """Parse SVG path data."""
    if not d_str:
        return []
    commands = []
    cmd = None
    params = []
    for token in _PATH_TOKEN.finditer(d_str):
        if token.group(1):
            if cmd:
                commands.append((cmd, params))
                params = []
            cmd = token.group(1)
        else:
            params.append(float(token.group(2)))
    if cmd:
        commands.append((cmd, params))
    return commands


def path_length(points):
    """Calculate path length."""
    length = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


def point_at_length(points, target_length):
    """Find point and angle at given length along path."""
    if not points:
        return (0, 0, 0)
    if len(points) == 1:
        return (points[0][0], points[0][1], 0)
    current_length = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if segment_length == 0:
            continue
        if current_length + segment_length >= target_length:
            # Point is on this segment
            t = (target_length - current_length) / segment_length
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            return (x, y, angle)
        current_length += segment_length
    # If we get here, return the last point
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return (x2, y2, angle)


def draw_path(c, commands, stroke_color=None, fill_color=None, stroke_width=1, dash=None):
    """Draw path on canvas."""
    if not commands:
        return
    c.saveState()
    if stroke_color and isinstance(stroke_color, tuple):
        c.setStrokeColorRGB(*stroke_color)
        c.setLineWidth(stroke_width)
        if dash:
            c.setDash(dash)
    if fill_color and isinstance(fill_color, tuple):
        if len(fill_color) == 3:
            c.setFillColorRGB(*fill_color)
        elif len(fill_color) == 4:
            c.setFillColorRGB(*fill_color[:3], alpha=fill_color[3])
    is_diamond = False
    points = []
    if len(commands) == 5:
        cmd_types = [cmd for cmd, _ in commands]
        if cmd_types[0] in ['M', 'm'] and cmd_types[-1] in ['Z', 'z'] and all(c_val in ['L', 'l'] for c_val in cmd_types[1:-1]):
            curr_x, curr_y = 0, 0
            for cmd, params in commands:
                if cmd == 'M':
                    curr_x, curr_y = params[0], params[1]
                    points.append((curr_x, curr_y))
                elif cmd == 'm':
                    curr_x += params[0]
                    curr_y += params[1]
                    points.append((curr_x, curr_y))
                elif cmd == 'L':
                    curr_x, curr_y = params[0], params[1]
                    points.append((curr_x, curr_y))
                elif cmd == 'l':
                    curr_x += params[0]
                    curr_y += params[1]
                    points.append((curr_x, curr_y))
            if len(points) == 4:
                is_diamond = True
    is_bracket = False
    if len(commands) == 2:
        cmd1, params1 = commands[0]
        cmd2, params2 = commands[1]
        if cmd1 == 'M' and cmd2 in ['L', 'V', 'H']:
            if cmd2 == 'V' or (cmd2 == 'L' and len(params2) >= 2 and abs(params1[0] - params2[0]) < 0.1):
                is_bracket = True
            elif cmd2 == 'H' or (cmd2 == 'L' and len(params2) >= 2 and abs(params1[1] - params2[1]) < 0.1 and abs(params1[0] - params2[0]) < 15):
                is_bracket = True
    if is_bracket:
        c.setLineJoin(0)
        c.setLineCap(0)
    else:
        c.setLineJoin(1)
        if stroke_width == 4.0 and not fill_color:
            c.setLineCap(1)
    if is_diamond:
        path = c.beginPath()
        for i, (x, y) in enumerate(points):
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.close()
    else:
        path = c.beginPath()
        curr_x, curr_y = 0.0, 0.0 # Ensure float for calculations
        start_x, start_y = None, None
        first_x, first_y = None, None
        last_cmd_processed = None
        for cmd, params in commands:
            if cmd == 'M':
                for i in range(0, len(params), 2):
                    curr_x, curr_y = float(params[i]), float(params[i+1])
                    if first_x is None: first_x, first_y = curr_x, curr_y
                    if start_x is None or i == 0: start_x, start_y = curr_x, curr_y
                    path.moveTo(curr_x, curr_y)
            elif cmd == 'm':
                for i in range(0, len(params), 2):
                    dx, dy = float(params[i]), float(params[i+1])
                    if start_x is None or i == 0: # First 'm' may be relative to an implicit (0,0) if path starts with 'm'
                        # Or if 'm' follows Z, curr_x, curr_y would be start_x, start_y of closed subpath
                        # For simplicity, assume curr_x, curr_y is correctly set from previous command or (0,0)
                        pass
                    curr_x += dx; curr_y += dy
                    if first_x is None: first_x, first_y = curr_x, curr_y
                    if start_x is None or i == 0: start_x, start_y = curr_x, curr_y
                    path.moveTo(curr_x, curr_y)
            elif cmd == 'L':
                for i in range(0, len(params), 2):
                    curr_x, curr_y = float(params[i]), float(params[i+1])
                    path.lineTo(curr_x, curr_y)
            elif cmd == 'l':
                for i in range(0, len(params), 2):
                    dx, dy = float(params[i]), float(params[i+1])
                    curr_x += dx; curr_y += dy
                    path.lineTo(curr_x, curr_y)
            elif cmd == 'H':
                for param in params:
                    curr_x = float(param); path.lineTo(curr_x, curr_y)
            elif cmd == 'h':
                for param in params:
                    curr_x += float(param); path.lineTo(curr_x, curr_y)
            elif cmd == 'V':
                for param in params:
                    curr_y = float(param); path.lineTo(curr_x, curr_y)
            elif cmd == 'v':
                for param in params:
                    curr_y += float(param); path.lineTo(curr_x, curr_y)
            elif cmd == 'Q': # Quadratic Bezier
                # A Q command has (x1 y1 x y)+ parameters. x1,y1 is control, x,y is endpoint.
                # Loop if multiple Q segments are chained (e.g., Q c1_1,e1_1 c1_2,e2_2)
                for i in range(0, len(params), 4):
                    x1, y1 = float(params[i]), float(params[i+1]) # control point
                    x, y = float(params[i+2]), float(params[i+3]) # end point
                    # Convert quadratic (x0,y0)-(x1,y1)-(x,y) to cubic for path.curveTo
                    # Current point (curr_x, curr_y) is x0, y0
                    c1x = curr_x + 2.0/3.0 * (x1 - curr_x)
                    c1y = curr_y + 2.0/3.0 * (y1 - curr_y)
                    c2x = x + 2.0/3.0 * (x1 - x)
                    c2y = y + 2.0/3.0 * (y1 - y)
                    path.curveTo(c1x, c1y, c2x, c2y, x, y)
                    curr_x, curr_y = x, y # Update current point
            elif cmd == 'q': # Relative Quadratic Bezier
                for i in range(0, len(params), 4):
                    dx1, dy1 = float(params[i]), float(params[i+1])
                    dx, dy = float(params[i+2]), float(params[i+3])
                    # Absolute control point
                    x1_abs = curr_x + dx1
                    y1_abs = curr_y + dy1
                    # Absolute end point
                    x_abs = curr_x + dx
                    y_abs = curr_y + dy
                    # Convert quadratic to cubic
                    c1x = curr_x + 2.0/3.0 * (x1_abs - curr_x)
                    c1y = curr_y + 2.0/3.0 * (y1_abs - curr_y)
                    c2x = x_abs + 2.0/3.0 * (x1_abs - x_abs)
                    c2y = y_abs + 2.0/3.0 * (y1_abs - y_abs)
                    path.curveTo(c1x, c1y, c2x, c2y, x_abs, y_abs)
                    curr_x, curr_y = x_abs, y_abs
            elif cmd == 'C': # Cubic Bezier
                for i in range(0, len(params), 6):
                    x1, y1 = float(params[i]), float(params[i+1])
                    x2, y2 = float(params[i+2]), float(params[i+3])
                    x, y = float(params[i+4]), float(params[i+5])
                    path.curveTo(x1, y1, x2, y2, x, y)
                    curr_x, curr_y = x, y
            elif cmd == 'c': # Relative Cubic Bezier
                for i in range(0, len(params), 6):
                    dx1, dy1 = float(params[i]), float(params[i+1])
                    dx2, dy2 = float(params[i+2]), float(params[i+3])
                    dx, dy = float(params[i+4]), float(params[i+5])
                    x1_abs, y1_abs = curr_x + dx1, curr_y + dy1
                    x2_abs, y2_abs = curr_x + dx2, curr_y + dy2
                    x_abs, y_abs = curr_x + dx, curr_y + dy
                    path.curveTo(x1_abs, y1_abs, x2_abs, y2_abs, x_abs, y_abs)
                    curr_x, curr_y = x_abs, y_abs
            # Note: 'S', 's' (smooth cubic), 'T', 't' (smooth quadratic) and 'A', 'a' (arc) are NOT handled here yet.
            # 'S' and 'T' rely on reflection of previous control points.
            # 'A' is complex to convert to Beziers.
            elif cmd == 'Z' or cmd == 'z':
                if start_x is not None and start_y is not None:
                    path.lineTo(start_x, start_y)
                path.close()
                if start_x is not None: curr_x, curr_y = start_x, start_y
            last_cmd_processed = cmd
        if first_x is not None and not dash and (last_cmd_processed != 'Z' and last_cmd_processed != 'z') and \
                sum(1 for cmd, _ in commands if cmd in ('M', 'm')) == 1 and \
                (abs(curr_x - first_x) > 1e-3 or abs(
                    curr_y - first_y) > 1e-3):  # Added tolerance for float comparison; a dashed open path must not be retraced, or the return stroke fills its own gaps
            path.lineTo(first_x, first_y)
            path.close()
    if fill_color and isinstance(fill_color, tuple):
        c.drawPath(path, fill=1, stroke=(stroke_color is not None and isinstance(stroke_color, tuple)))
    elif stroke_color and isinstance(stroke_color, tuple):
        c.drawPath(path, fill=0, stroke=1)
    c.restoreState()


def draw_rect(c, x, y, width, height, stroke_color=None, fill_color=None, stroke_width=1):
    """Draw rectangle on canvas."""
    c.saveState()
    c.setLineJoin(1)
    if stroke_color and isinstance(stroke_color, tuple):
        c.setStrokeColorRGB(*stroke_color)
        c.setLineWidth(stroke_width)
    if fill_color and isinstance(fill_color, tuple):
        if len(fill_color) == 3:
            c.setFillColorRGB(*fill_color)
        elif len(fill_color) == 4:
            c.setFillColorRGB(*fill_color[:3], alpha=fill_color[3])
    if fill_color and isinstance(fill_color, tuple) and stroke_color and isinstance(stroke_color, tuple):
        c.rect(x, y, width, height, fill=1, stroke=1)
    elif fill_color and isinstance(fill_color, tuple):
        c.rect(x, y, width, height, fill=1, stroke=0)
    elif stroke_color and isinstance(stroke_color, tuple):
        c.rect(x, y, width, height, fill=0, stroke=1)
    c.restoreState()


def draw_circle(c, cx, cy, r, stroke_color=None, fill_color=None, stroke_width=1):
    """Draw circle on canvas."""
    c.saveState()
    if stroke_color and isinstance(stroke_color, tuple):
        c.setStrokeColorRGB(*stroke_color)
        c.setLineWidth(stroke_width)
    if fill_color and isinstance(fill_color, tuple):
        if len(fill_color) == 3:
            c.setFillColorRGB(*fill_color)
        elif len(fill_color) == 4:
            c.setFillColorRGB(*fill_color[:3], alpha=fill_color[3])
    if fill_color and isinstance(fill_color, tuple) and stroke_color and isinstance(stroke_color, tuple):
        c.circle(cx, cy, r, fill=1, stroke=1)
    elif fill_color and isinstance(fill_color, tuple):
        c.circle(cx, cy, r, fill=1, stroke=0)
    elif stroke_color and isinstance(stroke_color, tuple):
        c.circle(cx, cy, r, fill=0, stroke=1)
    c.restoreState()


def draw_text_on_path(c, text, path_points, offset_percent, font_name, font_size, fill_color=None, text_anchor='start', offset_y=0, is_bold=False):
    """Draw text along path."""
    if not path_points or not text:
        return
    text = text.replace(' ', '')
    total_length = path_length(path_points)
    target_length = total_length * (offset_percent / 100.0)
    x, y, angle = point_at_length(path_points, target_length)
    if x == 0 and y == 0 and angle == 0:
        return
    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    if offset_y:
        c.translate(0, offset_y)
    if text_anchor == 'middle':
        text_width = pdfmetrics.stringWidth(text, font_name, font_size)
        c.translate(-text_width/2, 0)
    elif text_anchor == 'end':
        text_width = pdfmetrics.stringWidth(text, font_name, font_size)
        c.translate(-text_width, 0)
    if fill_color:
        c.setFillColorRGB(*fill_color)
    # Special handling for 'Df'
    if text == 'Df' and abs(offset_y - 0.5 * font_size) < 0.01:
        # Draw 'D' normally
        actual_font = font_name + '-Bold' if is_bold else font_name
        c.setFont(actual_font, font_size)
        c.scale(1, -1)
        d_width = pdfmetrics.stringWidth('D', actual_font, font_size)
        c.drawString(0, 0, 'D')
        # Draw 'f' with italic simulation
        c.saveState()
        c.translate(d_width, 0)
        c.transform(1, 0, 0.3, 1, -0.3 * font_size/2, 0)  # Positive skew for forward lean
        c.drawString(0, 0, 'f')
        c.restoreState()
    # Special handling for just 'f'
    elif (text == 'f') and abs(offset_y - 0.5 * font_size) < 0.01:
        # Simulate italic with proper forward slant
        actual_font = font_name + '-Bold' if is_bold else font_name
        c.setFont(actual_font, font_size)
        c.scale(1, -1)
        c.transform(1, 0, 0.3, 1, -0.3 * font_size/2, 0)  # Positive skew for forward lean
        c.drawString(0, 0, text)
    else:
        # Normal text rendering
        actual_font = font_name + '-Bold' if is_bold else font_name
        c.setFont(actual_font, font_size)
        c.scale(1, -1)
        c.drawString(0, 0, text, charSpace=1.0)
    c.restoreState()


def calculate_points(commands):
    """Calculate path points for layout."""
    points = []
    curr_x, curr_y = 0, 0
    for cmd, params in commands:
        if cmd == 'M':
            for i in range(0, len(params), 2):
                curr_x, curr_y = params[i], params[i+1]
                points.append((curr_x, curr_y))
        elif cmd == 'm':
            for i in range(0, len(params), 2):
                curr_x += params[i]
                curr_y += params[i+1]
                points.append((curr_x, curr_y))
        elif cmd == 'L':
            for i in range(0, len(params), 2):
                curr_x, curr_y = params[i], params[i+1]
                points.append((curr_x, curr_y))
        elif cmd == 'l':
            for i in range(0, len(params), 2):
                curr_x += params[i]
                curr_y += params[i+1]
                points.append((curr_x, curr_y))
        elif cmd == 'H':
            for param in params:
                curr_x = param
                points.append((curr_x, curr_y))
        elif cmd == 'h':
            for param in params:
                curr_x += param
                points.append((curr_x, curr_y))
        elif cmd == 'V':
            for param in params:
                curr_y = param
                points.append((curr_x, curr_y))
        elif cmd == 'v':
            for param in params:
                curr_y += param
                points.append((curr_x, curr_y))
        elif cmd == 'Z' or cmd == 'z':
            # Add closing point if needed
            if points and (points[0][0] != curr_x or points[0][1] != curr_y):
                points.append(points[0])  # Close the path properly
    return points


def _grad_num(el, key, default):
    raw = el.get(key)
    if raw is None:
        return default
    return float(raw.replace('%', '')) / 100 if '%' in raw else float(raw)


def _gradient_geometry(grad, bbox):
    """Gradient geometry in user space; objectBoundingBox fractions map onto the shape's box."""
    kind = grad.get('kind', 'radial')
    user_space = grad.get('units', 'objectBoundingBox') == 'userSpaceOnUse'
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if kind == 'linear':
        ax, ay, bx, by = grad.get('axis', [0.0, 0.0, 1.0, 0.0])
        if user_space:
            return kind, (ax, ay, bx, by)
        return kind, (x0 + w * ax, y0 + h * ay, x0 + w * bx, y0 + h * by)
    if user_space:
        return kind, (grad['cx'], grad['cy'], grad['r'])
    return kind, (x0 + w * grad['cx'], y0 + h * grad['cy'], grad['r'] * max(w, h))


def draw_radial_gradient_shape(c, cx, cy, r, stops, shape_func, geometry = None):
    """Fill a shape with a real radial gradient, which every backend resolves natively."""
    if not stops:
        return
    c.saveState()
    kind, coords = geometry or ('radial', (cx, cy, r))
    c.setFillGradient(kind, coords, stops)
    shape_func(c, cx, cy, r)
    c.restoreState()


def parse_svg_dimensions(root):
    """Parse SVG dimensions and viewbox."""
    width_raw = root.get('width', 100)
    height_raw = root.get('height', 100)
    # Convert to float, handling both numeric values and strings with units
    if isinstance(width_raw, (int, float)):
        width = float(width_raw)
    else:
        width = float(re.sub(r'[^\d.]', '', width_raw))
    if isinstance(height_raw, (int, float)):
        height = float(height_raw)
    else:
        height = float(re.sub(r'[^\d.]', '', height_raw))
    viewbox = root.get('viewBox', f'0 0 {width} {height}')
    vbox_parts = [float(x) for x in viewbox.split() if x.strip()]
    if len(vbox_parts) == 4:
        vb_x, vb_y, vb_width, vb_height = vbox_parts
    else:
        vb_x, vb_y, vb_width, vb_height = 0, 0, width, height
    scale_x = width / vb_width
    scale_y = height / vb_height
    return width, height, vb_x, vb_y, scale_x, scale_y


def _parse_inline_style(style_str: str) -> dict[str, str]: # Helper
    """Rudimentary parser for inline style attributes."""
    properties = {}
    if not style_str:
        return properties
    for item in style_str.strip().split(';'):
        if item and ':' in item:
            key, value = item.split(':', 1)
            properties[key.strip()] = value.strip()
    return properties


def _resolve_paint(elem, default_fill = 'auto', default_width = 1.0, use_opacity = False):
    """Resolve fill, stroke and stroke-width of an SVG element, honoring inline style, per-channel opacity and the general opacity attribute."""
    style = _parse_inline_style(elem.get('style', ''))
    raw_stroke = style.get('stroke', elem.get('stroke'))
    raw_fill = style.get('fill', elem.get('fill'))
    if raw_fill is None:
        raw_fill = ('black' if not raw_stroke else 'none') if default_fill == 'auto' else default_fill
    raw_width = style.get('stroke-width', elem.get('stroke-width'))
    try:
        width = float(str(raw_width).replace('px', '')) if raw_width else default_width
    except ValueError:
        width = default_width
    general = (style.get('opacity', elem.get('opacity')) if use_opacity else None) or None
    fill_val = parse_color(raw_fill)
    stroke_val = parse_color(raw_stroke if raw_stroke is not None else 'none')
    try:
        fill_alpha = float(style.get('fill-opacity', elem.get('fill-opacity')) or general or 1.0)
    except ValueError:
        fill_alpha = 1.0
    try:
        stroke_alpha = float(style.get('stroke-opacity', elem.get('stroke-opacity')) or general or 1.0)
    except ValueError:
        stroke_alpha = 1.0
    final_fill = fill_val[:3] + (min(fill_val[3], fill_alpha) if len(fill_val) == 4 else fill_alpha,) if isinstance(fill_val, tuple) else fill_val
    final_stroke = stroke_val[:3] + (min(stroke_val[3], stroke_alpha) if len(stroke_val) == 4 else stroke_alpha,) if isinstance(stroke_val, tuple) else None
    return final_fill, final_stroke, width


def _resolve_dash(elem):
    """Resolve stroke-dasharray of an SVG element into a dash pattern, honoring inline style."""
    raw = _parse_inline_style(elem.get('style', '')).get('stroke-dasharray', elem.get('stroke-dasharray'))
    if not raw or raw.strip() in ('none', ''):
        return None
    try:
        return [float(v) for v in re.split(r'[,\s]+', raw.strip()) if v]
    except ValueError:
        return None


def extract_defs(root, ns, elem_ctm = None):
    all_paths = {}
    all_gradients = {}
    for defs in root.findall('.//svg:defs', ns):
        for path in defs.findall('.//svg:path', ns):
            path_id = path.get('id', '')
            if not path_id:
                continue
            final_fill, final_stroke, stroke_width_val = _resolve_paint(path)
            path_commands = parse_path(path.get('d', ''))
            path_points = calculate_points(path_commands)
            all_paths[path_id] = {
                'points': path_points, 'commands': path_commands, 'dash': _resolve_dash(path),
                'stroke': final_stroke, 'fill': final_fill, 'stroke_width': stroke_width_val
            }
        for radial_gradient in list(defs.findall('.//svg:radialGradient', ns)) + list(
                defs.findall('.//svg:linearGradient', ns)):
            gradient_id = radial_gradient.get('id', '')
            if not gradient_id: continue
            kind = 'linear' if radial_gradient.tag.endswith('linearGradient') else 'radial'
            cx = float(radial_gradient.get('cx', '0.5').replace('%',''))/100 if '%' in radial_gradient.get('cx','0.5') else float(radial_gradient.get('cx','0.5'))
            cy = float(radial_gradient.get('cy', '0.5').replace('%',''))/100 if '%' in radial_gradient.get('cy','0.5') else float(radial_gradient.get('cy','0.5'))
            r_grad = float(radial_gradient.get('r', '0.5').replace('%',''))/100 if '%' in radial_gradient.get('r','0.5') else float(radial_gradient.get('r','0.5'))
            stops = []
            for stop in radial_gradient.findall('.//svg:stop', ns):
                offset = float(stop.get('offset', '0').replace('%',''))/100 if '%' in stop.get('offset','0') else float(stop.get('offset','0'))
                stop_color_str = stop.get('stop-color', 'white')
                opacity = float(stop.get('stop-opacity', '1'))
                color_tuple = None
                style_props_stop = _parse_inline_style(stop.get('style',''))
                if 'stop-color' in style_props_stop: stop_color_str = style_props_stop['stop-color']
                if 'stop-opacity' in style_props_stop:
                    try: opacity = float(style_props_stop['stop-opacity'])
                    except ValueError: pass
                parsed_stop_color = parse_color(stop_color_str)
                if parsed_stop_color and isinstance(parsed_stop_color, tuple):
                    color_tuple = parsed_stop_color + (opacity,)
                elif stop_color_str.startswith('#'):
                    if len(stop_color_str) == 7:
                        r_val = int(stop_color_str[1:3],16)/255.0; g_val=int(stop_color_str[3:5],16)/255.0; b_val=int(stop_color_str[5:7],16)/255.0
                        color_tuple = (r_val,g_val,b_val,opacity)
                if color_tuple: stops.append((offset, color_tuple))
            axis = [_grad_num(radial_gradient, k, dv) for k, dv in (('x1', 0.0), ('y1', 0.0), ('x2', 1.0), ('y2', 0.0))]
            all_gradients[gradient_id] = {'kind': kind, 'cx': cx, 'cy': cy, 'r': r_grad, 'axis': axis, 'stops': stops,
                                          'units': radial_gradient.get('gradientUnits', 'objectBoundingBox')}
    in_defs = {id(node) for defs_node_check in root.findall('.//svg:defs', ns) for node in defs_node_check.iter()}
    for path in root.findall('.//svg:path', ns):  # Connection path logic
        # RDKit bonds usually defined by class, e.g. "bond-0" and explicit style.
        if 'snfg-linkage' in (path.get('class') or '') or path.get('stroke-width') == '4.0':
            if id(path) in in_defs: continue
            path_id = path.get('id') or f"connection_{len(all_paths)}"
            path.set('id', path_id)
            final_fill_conn, final_stroke_conn, stroke_width_val_conn = _resolve_paint(path, default_width = 4.0)
            path_commands = parse_path(path.get('d', ''))
            path_points = calculate_points(path_commands)
            all_paths[path_id] = {
                'points': path_points, 'commands': path_commands, 'is_connection': True, 'dash': _resolve_dash(path),
                'ctm': elem_ctm.get(id(path), _IDENTITY) if elem_ctm is not None else _IDENTITY,
                'stroke': final_stroke_conn, 'fill': final_fill_conn, 'stroke_width': stroke_width_val_conn
            }
    return all_paths, all_gradients


def draw_ellipse(c, cx, cy, rx, ry, stroke_color=None, fill_color=None, stroke_width=1):
    c.saveState()
    if stroke_color and isinstance(stroke_color, tuple):
        c.setStrokeColorRGB(*stroke_color)
        c.setLineWidth(stroke_width)
    if fill_color and isinstance(fill_color, tuple):
        if len(fill_color) == 3:
            c.setFillColorRGB(*fill_color)
        elif len(fill_color) == 4:
            c.setFillColorRGB(*fill_color[:3], alpha=fill_color[3])
    x = cx - rx
    y = cy - ry
    width = 2 * rx
    height = 2 * ry
    do_fill = fill_color and isinstance(fill_color, tuple)
    do_stroke = stroke_color and isinstance(stroke_color, tuple)
    if do_fill or do_stroke:
        c.ellipse(x, y, x + width, y + height, fill=1 if do_fill else 0, stroke=1 if do_stroke else 0)
    c.restoreState()


_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _apply_ctm(c, elem_ctm, el):
    """Push the element's accumulated ancestor transform; caller restores."""
    c.saveState()
    if elem_ctm is not None:
        c.transform(*elem_ctm.get(id(el), _IDENTITY))


def element_transforms(root):
    """Accumulated transform for every element, so nested <g transform=...> is honoured.

    glycowork wraps brackets and Domon-Costello Z/Y markers in rotated groups; without this
    their children were drawn unrotated.
    """
    out, stack = {}, [(root, _IDENTITY)]
    while stack:
        el, parent = stack.pop()
        m = svgin._mul(svgin._transform(el.get('transform')), parent)
        out[id(el)] = m
        stack.extend((child, m) for child in el)
    return out


def find_connection_paths(root, all_paths, ns, elem_ctm = None):
    """Find connection paths which are used as lines between elements."""
    connection_path_ids = set()
    for use_elem in root.findall('.//svg:use', ns):
        href = use_elem.get('{http://www.w3.org/1999/xlink}href')
        if not href:
            href = use_elem.get('href')
        if not href or not href.startswith('#'):
            continue
        path_id = href[1:]
        if path_id in all_paths:
            # Check if this is likely a connection path by looking at its properties
            path_info = all_paths[path_id]
            if elem_ctm is not None:  # a <use> places the def, so its transform is the one that counts
                path_info['ctm'] = elem_ctm.get(id(use_elem), _IDENTITY)
            # Connection paths typically have a stroke but no fill
            if path_info['stroke'] and not path_info['fill']:
                connection_path_ids.add(path_id)
    for path_id, path_info in all_paths.items():
        # Identify connection paths (inline linkage lines harvested by extract_defs)
        if path_info.get('is_connection'):
            connection_path_ids.add(path_id)
    return connection_path_ids


def draw_circles_with_gradients(c, root, all_gradients, ns, elem_ctm = None):
    """Draw circles with gradient fills."""
    for circle in root.findall('.//svg:circle', ns):
        _apply_ctm(c, elem_ctm, circle)
        cx = float(circle.get('cx', '0'))
        cy = float(circle.get('cy', '0'))
        r = float(circle.get('r', '0'))
        fill = parse_color(circle.get('fill', 'none'))
        if not (isinstance(fill, str) and fill in all_gradients):
            c.restoreState()
            continue
        grad = all_gradients[fill]
        draw_radial_gradient_shape(c, cx, cy, r, grad['stops'],
                                   lambda canvas, center_x, center_y, radius: canvas.circle(center_x, center_y,
                                                                                            radius, fill = 1,
                                                                                            stroke = 0),
                                   _gradient_geometry(grad, (cx - r, cy - r, cx + r, cy + r)))
        c.restoreState()


def draw_connection_paths(c, connection_path_ids, all_paths, root=None, ns=None):
    """Draw connection paths between elements."""
    c.setLineCap(1)  # Set round cap for all connection lines
    for path_id in connection_path_ids:
        path_info = all_paths[path_id]
        c.saveState()
        c.transform(*path_info.get('ctm', _IDENTITY))
        commands = path_info['commands'][:]  # Copy to avoid modifying original
        # Check if this path ends at an invisible circle and shorten if needed
        if root is not None and ns is not None and commands:
            commands = shorten_if_invisible_endpoint(commands, root, ns)
        draw_path(c, commands, path_info['stroke'], None, path_info['stroke_width'], dash = path_info.get('dash'))
        c.restoreState()

def shorten_if_invisible_endpoint(commands, root, ns):
    """Shorten connection path if it ends at an invisible circle."""
    if len(commands) < 2:
        return commands
    # Get endpoint from last command
    last_cmd, last_params = commands[-1]
    if last_cmd != 'L' or len(last_params) < 2:
        return commands
    end_x, end_y = last_params[0], last_params[1]
    # Check for invisible circle at endpoint
    for circle in root.findall('.//svg:circle', ns):
        cx = float(circle.get('cx', '0'))
        cy = float(circle.get('cy', '0'))
        if abs(cx - end_x) < 1 and abs(cy - end_y) < 1:  # Same position
            fill = circle.get('fill', '')
            stroke = circle.get('stroke', '')
            if fill == 'none' and (stroke == 'none' or stroke == ''):
                # Invisible circle found, shorten the path
                first_cmd, first_params = commands[0]
                if first_cmd == 'M' and len(first_params) >= 2:
                    start_x, start_y = first_params[0], first_params[1]
                    dx = end_x - start_x
                    dy = end_y - start_y
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 25:  # Shorten by 25 units
                        factor = (length - 25) / length
                        new_end_x = start_x + dx * factor
                        new_end_y = start_y + dy * factor
                        commands[-1] = (last_cmd, [new_end_x, new_end_y])
                break
    return commands


def draw_direct_text(c, text, x, y, font_to_use, font_size, fill_color=None, text_anchor='start'):
    """Draw text at specified coordinates."""
    c.saveState()
    # Handle text anchor positioning
    if text_anchor == 'middle':
        text_width = pdfmetrics.stringWidth(text, font_to_use, font_size)
        x -= text_width/2
    elif text_anchor == 'end':
        text_width = pdfmetrics.stringWidth(text, font_to_use, font_size)
        x -= text_width
    if fill_color:
        c.setFillColorRGB(*fill_color)
    # Since we've already flipped the canvas (scale 1, -1), we need to flip back
    # for the text to be right side up
    c.translate(x, y)
    c.scale(1, -1)
    # Now set the font and draw at the origin (0,0) since we've translated
    c.setFont(font_to_use, font_size)
    c.drawString(0, 0, text)
    c.restoreState()


def process_text_elements(c, root, all_paths, ns, font_to_use, elem_ctm = None):
    """Process and draw text elements including text on paths."""
    for text in root.findall('.//svg:text', ns):
        _apply_ctm(c, elem_ctm, text)
        font_size = float(text.get('font-size', '12'))
        fill = parse_color(text.get('fill', '#000000'))
        text_anchor = text.get('text-anchor', 'start')
        # Check if this is a direct text element with x and y attributes
        x = text.get('x')
        y = text.get('y')
        if x is not None and y is not None and text.text:
            # This is a direct text element (not on a path)
            x = float(x)
            y = float(y)
            text_content = text.text.strip()
            if text_content:
                draw_direct_text(c, text_content, x, y, font_to_use, font_size, fill, text_anchor)
            c.restoreState()
            continue
        for textpath in text.findall('.//svg:textPath', ns):
            href = textpath.get('{http://www.w3.org/1999/xlink}href')
            if not href:
                href = textpath.get('href')
            if not href or not href.startswith('#'):
                continue
            path_id = href[1:]
            if path_id not in all_paths:
                continue
            path_points = all_paths[path_id]['points']
            text_content = ""
            offset_y = 0
            is_bold = False
            for tspan in textpath.findall('.//svg:tspan', ns):
                if tspan.text:
                    text_content += tspan.text
                    dy = tspan.get('dy', '')
                    if dy and 'em' in dy:
                        try:
                            em_value = float(dy.replace('em', ''))
                            offset_y = em_value * font_size
                            if abs(em_value + 3.15) < 0.01:
                                is_bold = True
                        except ValueError:
                            pass
            if not text_content and textpath.text:
                text_content = textpath.text
            start_offset = textpath.get('startOffset', '50%')
            offset_percent = 50
            if start_offset.endswith('%'):
                offset_percent = float(start_offset[:-1])
            elif start_offset.isdigit():
                offset_percent = float(start_offset) / path_length(path_points) * 100
            draw_text_on_path(c, text_content, path_points, offset_percent,
                              font_to_use, font_size, fill, text_anchor, offset_y = offset_y, is_bold = is_bold)
        c.restoreState()


def register_bundled_fonts():
    """Register bundled Comfortaa font, or Century Gothic, if available."""
    # Common Century Gothic filenames across platforms
    century_gothic_variations = [
        # Windows standard names
        ('GOTHIC.TTF', 'GOTHICB.TTF'),
        ('gothic.ttf', 'gothicb.ttf'),
        # macOS/Linux possible names
        ('Century Gothic.ttf', 'Century Gothic Bold.ttf'),
        ('CenturyGothic.ttf', 'CenturyGothic-Bold.ttf'),
        ('CenturyGothic-Regular.ttf', 'CenturyGothic-Bold.ttf'),
        # Other variations
        ('century_gothic.ttf', 'century_gothic_bold.ttf'),
        ('CenturyGothic.ttf', 'CenturyGothicBold.ttf')
    ]
    # Try to register Century Gothic with various filenames
    for regular_name, bold_name in century_gothic_variations:
        try:
            # Try with just the filename (reportlab will find it in system fonts)
            pdfmetrics.registerFont(TTFont('CenturyGothic', regular_name))
            pdfmetrics.registerFont(TTFont('CenturyGothic-Bold', bold_name))
            pdfmetrics.registerFontFamily('CenturyGothic', normal='CenturyGothic', bold='CenturyGothic-Bold')
            return 'CenturyGothic'
        except Exception:
            continue  # try the next filename variation
    font_name = 'Comfortaa'
    # Get the location of this module file and navigate to fonts directory
    this_dir = Path(__file__).parent / 'fonts'
    font_regular = this_dir / 'Comfortaa-Regular.ttf'
    font_bold = this_dir / 'Comfortaa-Bold.ttf'
    pdfmetrics.registerFont(TTFont(font_name, str(font_regular)))
    pdfmetrics.registerFont(TTFont(f'{font_name}-Bold', str(font_bold)))
    pdfmetrics.registerFontFamily(font_name, normal=font_name, bold=f'{font_name}-Bold')
    return font_name


# Register bundled font
font_to_use = register_bundled_fonts()


def _render_svg_to_pdf_canvas(svg_data: str,
                              pdf_target: Union[str, Path, BytesIO],
                              alt_text_info: Union[dict, None] = None) -> canvas.Canvas:
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    current_alt_text = None
    if alt_text_info and 'alt_text' in alt_text_info:
        current_alt_text = alt_text_info['alt_text']
    else:
        aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
        if aria_label_match:
            current_alt_text = aria_label_match.group(1)
    root = ET.fromstring(svg_data)
    ns = {'svg': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'}
    width, height, vb_x, vb_y, scale_x, scale_y = parse_svg_dimensions(root)
    c = canvas.Canvas(pdf_target, pagesize=(width * mm, height * mm) if width <= 20 and height <=20 else (width, height))
    if current_alt_text:
        c.setTitle(current_alt_text.replace("SNFG diagram of ", "").split(" drawn in")[0])
        c.setAuthor("GlycoDraw")
        c.setSubject("Glycan Visualization")
        c.setKeywords(f"glycan;carbohydrate;glycowork;Description: {current_alt_text}")
    elem_ctm = element_transforms(root)
    all_paths, all_gradients = extract_defs(root, ns, elem_ctm)
    connection_path_ids = find_connection_paths(root, all_paths, ns, elem_ctm)
    c.translate(0, height)
    c.scale(1, -1)
    c.translate(-vb_x * scale_x, -vb_y * scale_y)
    c.scale(scale_x, scale_y)
    draw_circles_with_gradients(c, root, all_gradients, ns, elem_ctm)
    draw_connection_paths(c, connection_path_ids, all_paths, root, ns)
    for circle_element in root.findall('.//svg:circle', ns):
        _apply_ctm(c, elem_ctm, circle_element)
        final_fill_c, final_stroke_c, sw_c = _resolve_paint(circle_element)
        is_gradient_fill = isinstance(final_fill_c, str) and final_fill_c in all_gradients
        if is_gradient_fill and final_stroke_c is None:  # Gradient and no separate stroke: already handled by draw_circles_with_gradients
            c.restoreState()
            continue
        cx_c = float(circle_element.get('cx', '0'));
        cy_c = float(circle_element.get('cy', '0'));
        r_c = float(circle_element.get('r', '0'))
        draw_circle(c, cx_c, cy_c, r_c, final_stroke_c, None if isinstance(final_fill_c, str) else final_fill_c, sw_c)
        c.restoreState()
    for rect_element in root.findall('.//svg:rect', ns):
        _apply_ctm(c, elem_ctm, rect_element)
        final_fill_r, final_stroke_r, sw_r = _resolve_paint(rect_element, default_fill = 'none', use_opacity = True)
        x_r = float(rect_element.get('x', '0')); y_r = float(rect_element.get('y', '0'))
        w_r = float(rect_element.get('width', '0')); h_r = float(rect_element.get('height', '0'))
        if isinstance(final_fill_r, str) and final_fill_r in all_gradients and all_gradients[final_fill_r]['stops']:
            grad_r_data = all_gradients[final_fill_r]
            c.saveState()
            c.setFillGradient(*_gradient_geometry(grad_r_data, (x_r, y_r, x_r + w_r, y_r + h_r)), grad_r_data['stops'])
            draw_rect(c, x_r, y_r, w_r, h_r, final_stroke_r, grad_r_data['stops'][0][1][:3], sw_r)
            c.restoreState()
        else:
            draw_rect(c, x_r, y_r, w_r, h_r, final_stroke_r, final_fill_r, sw_r)
        c.restoreState()
    for ellipse_element in root.findall('.//svg:ellipse', ns):
        _apply_ctm(c, elem_ctm, ellipse_element)
        final_fill_e, final_stroke_e, sw_e = _resolve_paint(ellipse_element, default_fill = 'black')
        cx_e = float(ellipse_element.get('cx', '0')); cy_e = float(ellipse_element.get('cy', '0'))
        rx_e = float(ellipse_element.get('rx', '0')); ry_e = float(ellipse_element.get('ry', '0'))
        if isinstance(final_fill_e, str) and final_fill_e in all_gradients and all_gradients[final_fill_e]['stops']:
            grad_e_data = all_gradients[final_fill_e]
            c.saveState()
            c.setFillGradient(*_gradient_geometry(grad_e_data, (cx_e - rx_e, cy_e - ry_e, cx_e + rx_e, cy_e + ry_e)),
                              grad_e_data['stops'])
            draw_ellipse(c, cx_e, cy_e, rx_e, ry_e, final_stroke_e, grad_e_data['stops'][0][1][:3], sw_e)
            c.restoreState()
        else:
            draw_ellipse(c, cx_e, cy_e, rx_e, ry_e, final_stroke_e, final_fill_e, sw_e)
        c.restoreState()
    for path_element in root.findall('.//svg:path', ns):
        _apply_ctm(c, elem_ctm, path_element)
        is_in_defs_p = any(path_element in list(defs_el_iter) for defs_el_iter in root.findall('.//svg:defs', ns))
        path_id_p = path_element.get('id', '')
        path_data_p = path_element.get('d', '')
        if is_in_defs_p or path_id_p in connection_path_ids or not path_data_p:
            c.restoreState()
            continue
        final_fill_p, final_stroke_p, sw_p = _resolve_paint(path_element)
        path_commands_p = parse_path(path_data_p)
        dash_p = _resolve_dash(path_element)
        if isinstance(final_fill_p, str) and final_fill_p in all_gradients and all_gradients[final_fill_p]['stops']:
            grad_p = all_gradients[final_fill_p]
            first_stop_p = grad_p['stops'][0][1]
            pts_p = calculate_points(path_commands_p) or [(0, 0)]
            bbox_p = (min(p[0] for p in pts_p), min(p[1] for p in pts_p), max(p[0] for p in pts_p),
                      max(p[1] for p in pts_p))
            c.saveState()
            c.setFillGradient(*_gradient_geometry(grad_p, bbox_p), grad_p['stops'])
            draw_path(c, path_commands_p, final_stroke_p,
                      first_stop_p[:3] + (first_stop_p[3] if len(first_stop_p) > 3 else 1.0,), sw_p, dash_p)
            c.restoreState()
        else:
            draw_path(c, path_commands_p, final_stroke_p, final_fill_p, sw_p, dash_p)
        c.restoreState()
    process_text_elements(c, root, all_paths, ns, font_to_use, elem_ctm)
    return c


def convert_chem_to_file(svg_data: str, file_path: Union[str, Path, None] = None, return_bytes: bool = False):
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    ext = 'png' if file_path is None else str(file_path).lower().split('.')[-1]
    if ext not in ('png', 'pdf'):
        raise ValueError(f"Unsupported extension: {ext}")
    target = BytesIO() if (ext == 'pdf' and return_bytes) else (file_path if ext == 'pdf' else None)
    canvas_obj = _render_svg_to_pdf_canvas(svg_data, target, alt_text_info = None)
    if ext == 'pdf':
        canvas_obj.save()
        if return_bytes:
            data = target.getvalue()
            target.close()
            return data
        return None
    png = canvas_obj.to_png(300 / 72.0, 300 / 72.0)  # RDKit path renders at a fixed 300 dpi
    if return_bytes:
        return png
    if file_path is None:
        raise ValueError("file_path must be provided for PNG output if not returning bytes.")
    with open(str(file_path), 'wb') as fh:
        fh.write(png)
    return None


def convert_svg_to_pdf(svg_data: str, pdf_file_path: Union[str, Path], return_canvas: bool = False, chem: bool = False,
                       shadow: bool = False):
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    if chem:
        convert_chem_to_file(svg_data, file_path = pdf_file_path, return_bytes = False)
        return None
    alt_text_payload = None
    aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
    if aria_label_match:
        alt_text_payload = {'alt_text': aria_label_match.group(1)}
    canvas_obj = _render_svg_to_pdf_canvas(svg_data, pdf_file_path, alt_text_info = alt_text_payload)
    canvas_obj.shadow = shadow
    if return_canvas:
        return canvas_obj
    else:
        canvas_obj.save()
        return None


def convert_svg_to_png(svg_data: str, png_file_path: Union[str, Path, None] = None,
                       output_width: Union[int, None] = None, output_height: Union[int, None] = None,
                       scale: Union[float, None] = None, return_bytes: bool = False,
                       chem: bool = False, background: Union[tuple, None] = None,
                       shadow: bool = False):
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    if chem:
        return convert_chem_to_file(svg_data, file_path = png_file_path, return_bytes = return_bytes)
    if not return_bytes and png_file_path is None:
        raise ValueError("png_file_path must be provided if return_bytes is False.")
    aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
    alt_text = aria_label_match.group(1) if aria_label_match else None
    canvas_obj = _render_svg_to_pdf_canvas(svg_data, None, alt_text_info = {'alt_text': alt_text} if alt_text else None)
    canvas_obj.shadow = shadow
    page_width = canvas_obj.width if canvas_obj.width > 1e-3 else 1.0
    page_height = canvas_obj.height if canvas_obj.height > 1e-3 else 1.0
    if scale is not None:
        zoom_x = zoom_y = scale
    elif output_width is not None and output_height is not None:
        zoom_x, zoom_y = output_width / page_width, output_height / page_height
    elif output_width is not None:
        zoom_x = zoom_y = output_width / page_width
    elif output_height is not None:
        zoom_x = zoom_y = output_height / page_height
    else:
        zoom_x = zoom_y = 1.0
    png = canvas_obj.to_png(zoom_x, zoom_y, background, texts = (('alt', alt_text),) if alt_text else ())
    if return_bytes:
        return png
    with open(str(png_file_path), 'wb') as fh:
        fh.write(png)
    return None


def pdf_to_svg_bytes(svg_string):
    """Round-trip an SNFG SVG through the glycorender display list so the browser sees the PDF geometry."""
    return _render_svg_to_pdf_canvas(svg_string, None, alt_text_info=None).to_svg()


def simple_svg_to_pdf(svg_data: str, pdf_path: Union[str, Path]) -> None:
    """Convert composite SVG to PDF without additional glycan-specific rendering."""
    svgin.build(svg_data, str(pdf_path), (font_to_use, font_to_use + '-Bold')).save()


def simple_svg_to_png(svg_data: str, png_path: Union[str, Path]) -> None:
    """Convert composite SVG to PNG without additional glycan-specific rendering."""
    with open(str(png_path), 'wb') as fh:
        fh.write(svgin.build(svg_data, None, (font_to_use, font_to_use + '-Bold')).to_png(300 / 72.0, 300 / 72.0))