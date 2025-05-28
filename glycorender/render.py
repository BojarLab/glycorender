import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
import xml.etree.ElementTree as ET
import tempfile
import re
import os
import math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
from io import BytesIO
from PIL import Image, PngImagePlugin
from typing import Union, List, Tuple, Any, Dict

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
    if len(color_str) == 7:  # #RRGGBB
      if color_str == "#000000":
        return reportlab_colors['charcoal']
      r = int(color_str[1:3], 16) / 255.0
      g = int(color_str[3:5], 16) / 255.0
      b = int(color_str[5:7], 16) / 255.0
      return (r, g, b)
  if color_str.startswith('url(#'):
    # Return the gradient ID
    return color_str[5:-1]  # Remove 'url(#' and ')'
  if color_str in reportlab_colors:
    return reportlab_colors[color_str]
  return reportlab_colors['charcoal']


def parse_path(d_str):
  """Parse SVG path data."""
  if not d_str:
    return []
  tokens = []
  i = 0
  d_len = len(d_str)
  while i < d_len:
    if d_str[i].isalpha():
      tokens.append(d_str[i])
      i += 1
    elif d_str[i].isdigit() or d_str[i] in '.-':
      num_start = i
      i += 1
      while i < d_len and (d_str[i].isdigit() or d_str[i] == '.'):
        i += 1
      tokens.append(float(d_str[num_start:i]))
    else:
      i += 1
  # Process tokens into commands
  commands = []
  cmd = None
  params = []
  for token in tokens:
    if isinstance(token, str):
      if cmd:
        commands.append((cmd, params))
        params = []
      cmd = token
    else:
      params.append(token)
  if cmd and params:
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
        return (0, 0, 0)
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


def draw_path(c, commands, stroke_color=None, fill_color=None, stroke_width=1):
  """Draw path on canvas."""
  if not commands:
    return
  c.saveState()
  if stroke_color and isinstance(stroke_color, tuple):
    c.setStrokeColorRGB(*stroke_color)
    c.setLineWidth(stroke_width)
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
    if first_x is not None and (last_cmd_processed != 'Z' and last_cmd_processed != 'z') and \
       (abs(curr_x - first_x) > 1e-3 or abs(curr_y - first_y) > 1e-3): # Added tolerance for float comparison
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


def process_use_element(c, use_elem, all_paths, ns):
  """Process a use element which refers to a path."""
  href = use_elem.get('{http://www.w3.org/1999/xlink}href')
  if not href:
    href = use_elem.get('href')
  if not href or not href.startswith('#'):
    return
  path_id = href[1:]
  if path_id not in all_paths:
    return
  path_info = all_paths[path_id]
  # Draw the path with stroke and no fill to ensure we see the line
  draw_path(c, path_info['commands'], path_info['stroke'], None, path_info['stroke_width'])


def draw_radial_gradient_shape(c, cx, cy, r, stops, shape_func):
  """Draw a radial gradient with extremely smooth appearance."""
  # Sort stops by offset
  stops = sorted(stops, key=lambda x: x[0])
  # First draw base with innermost color
  innermost_color = stops[-1][1]
  c.saveState()
  c.setFillColorRGB(*innermost_color[:3], alpha=innermost_color[3] * 0.3)
  shape_func(c, cx, cy, r)
  c.restoreState()
  # Large number of steps with linear spacing for smooth transition
  # Adjust based on radius - more circles for bigger radii
  num_steps = max(30, min(60, int(r * 1.2)))
  # Store the max offset for proper scaling
  max_offset = stops[-1][0]
  # Create circles from largest to smallest
  for i in range(num_steps):
    # Linear position in 0-1 range
    t = i / float(num_steps - 1)
    # Calculate radius with very small changes between circles
    # Keep slight spacing at center to avoid overcrowding
    radius = r * (1.0 - 0.98 * t)
    # Convert t to gradient position
    gradient_pos = (1.0 - t) * max_offset
    # Find segment in gradient this belongs to
    for j in range(len(stops) - 1):
      start_offset, start_color = stops[j]
      end_offset, end_color = stops[j + 1]
      if start_offset <= gradient_pos <= end_offset:
        # Calculate position within this segment
        segment_t = (gradient_pos - start_offset) / (end_offset - start_offset)
        # Linear color interpolation
        r_val = start_color[0] + (end_color[0] - start_color[0]) * segment_t
        g_val = start_color[1] + (end_color[1] - start_color[1]) * segment_t
        b_val = start_color[2] + (end_color[2] - start_color[2]) * segment_t
        # Calculate very subtle opacity changes between adjacent circles
        base_alpha = start_color[3] + (end_color[3] - start_color[3]) * segment_t
        # Use gentle opacity curve for visually smooth transition
        opacity = base_alpha * 0.3 * (0.2 + 0.8 * (1.0 - t))
        # Only draw if visible
        if radius > 0 and opacity > 0.0005:
          c.saveState()
          c.setFillColorRGB(r_val, g_val, b_val, alpha=opacity)
          shape_func(c, cx, cy, radius)
          c.restoreState()
        break


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


def _parse_inline_style(style_str: str) -> Dict[str, str]: # Helper
    """Rudimentary parser for inline style attributes."""
    properties = {}
    if not style_str:
        return properties
    for item in style_str.strip().split(';'):
        if item and ':' in item:
            key, value = item.split(':', 1)
            properties[key.strip()] = value.strip()
    return properties


def extract_defs(root, ns):
  all_paths = {}
  all_gradients = {}
  for defs in root.findall('.//svg:defs', ns):
    for path in defs.findall('.//svg:path', ns):
      path_id = path.get('id', '')
      if not path_id:
        continue
      path_data = path.get('d', '')
      style_props = _parse_inline_style(path.get('style', ''))
      raw_fill = style_props.get('fill', path.get('fill'))
      raw_stroke = style_props.get('stroke', path.get('stroke'))
      raw_stroke_width = style_props.get('stroke-width', path.get('stroke-width', '1'))
      raw_fill_opacity = style_props.get('fill-opacity', path.get('fill-opacity')) # Added
      raw_stroke_opacity = style_props.get('stroke-opacity', path.get('stroke-opacity')) # Added
      fill_val = parse_color(raw_fill if raw_fill is not None else ('black' if not raw_stroke and not style_props.get('stroke') else 'none'))
      stroke_val = parse_color(raw_stroke if raw_stroke is not None else 'none')
      try:
          stroke_width_val = float(str(raw_stroke_width).replace('px','')) if raw_stroke_width else 1.0
      except ValueError:
          stroke_width_val = 1.0
      final_fill = None # Process opacity
      if fill_val and isinstance(fill_val, tuple):
          alpha_f = 1.0
          if raw_fill_opacity:
              try: alpha_f = float(raw_fill_opacity)
              except ValueError: pass
          final_fill = fill_val + (alpha_f,) if len(fill_val) == 3 else (fill_val[0],fill_val[1],fill_val[2], min(fill_val[3], alpha_f) if len(fill_val) == 4 else alpha_f)
      elif isinstance(fill_val, str): final_fill = fill_val
      final_stroke = None # Process opacity
      if stroke_val and isinstance(stroke_val, tuple):
          alpha_s = 1.0
          if raw_stroke_opacity:
              try: alpha_s = float(raw_stroke_opacity)
              except ValueError: pass
          final_stroke = stroke_val + (alpha_s,) if len(stroke_val) == 3 else (stroke_val[0],stroke_val[1],stroke_val[2], min(stroke_val[3], alpha_s) if len(stroke_val) == 4 else alpha_s)
      path_commands = parse_path(path_data)
      path_points = calculate_points(path_commands)
      all_paths[path_id] = {
        'points': path_points, 'commands': path_commands,
        'stroke': final_stroke, 'fill': final_fill, 'stroke_width': stroke_width_val
      }
    for radial_gradient in defs.findall('.//svg:radialGradient', ns):
      gradient_id = radial_gradient.get('id', '')
      if not gradient_id: continue
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
      all_gradients[gradient_id] = {'cx': cx, 'cy': cy, 'r': r_grad,'stops': stops}
  for path in root.findall('.//svg:path', ns): # Connection path logic
    # RDKit bonds usually defined by class, e.g. "bond-0" and explicit style.
    if path.get('stroke-width') == '4.0':
      is_in_defs_check = any(path in list(defs_node_check) for defs_node_check in root.findall('.//svg:defs', ns))
      if is_in_defs_check: continue
      path_id = path.get('id', f"connection_{len(all_paths)}")
      path_data = path.get('d', '')
      style_props_conn = _parse_inline_style(path.get('style', ''))
      raw_fill_conn = style_props_conn.get('fill', path.get('fill'))
      raw_stroke_conn = style_props_conn.get('stroke', path.get('stroke'))
      raw_stroke_width_conn = style_props_conn.get('stroke-width', path.get('stroke-width'))
      raw_fill_opacity_conn = style_props_conn.get('fill-opacity', path.get('fill-opacity'))
      raw_stroke_opacity_conn = style_props_conn.get('stroke-opacity', path.get('stroke-opacity'))
      fill_val_conn = parse_color(raw_fill_conn if raw_fill_conn is not None else ('black' if not raw_stroke_conn and not style_props_conn.get('stroke') else 'none'))
      stroke_val_conn = parse_color(raw_stroke_conn if raw_stroke_conn is not None else 'none')
      try:
          stroke_width_val_conn = float(str(raw_stroke_width_conn).replace('px','')) if raw_stroke_width_conn else 4.0
      except ValueError: stroke_width_val_conn = 4.0
      final_fill_conn = None
      if fill_val_conn and isinstance(fill_val_conn, tuple):
          alpha_f_conn = 1.0
          if raw_fill_opacity_conn:
              try: alpha_f_conn = float(raw_fill_opacity_conn)
              except ValueError: pass
          final_fill_conn = fill_val_conn + (alpha_f_conn,) if len(fill_val_conn) == 3 else (fill_val_conn[0],fill_val_conn[1],fill_val_conn[2],min(fill_val_conn[3], alpha_f_conn) if len(fill_val_conn) == 4 else alpha_f_conn)
      elif isinstance(fill_val_conn, str): final_fill_conn = fill_val_conn
      final_stroke_conn = None
      if stroke_val_conn and isinstance(stroke_val_conn, tuple):
          alpha_s_conn = 1.0
          if raw_stroke_opacity_conn:
              try: alpha_s_conn = float(raw_stroke_opacity_conn)
              except ValueError: pass
          final_stroke_conn = stroke_val_conn + (alpha_s_conn,) if len(stroke_val_conn) == 3 else (stroke_val_conn[0],stroke_val_conn[1],stroke_val_conn[2],min(stroke_val_conn[3], alpha_s_conn) if len(stroke_val_conn) == 4 else alpha_s_conn)
      path_commands = parse_path(path_data)
      path_points = calculate_points(path_commands)
      all_paths[path_id] = {
        'points': path_points, 'commands': path_commands,
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


def find_connection_paths(root, all_paths, ns):
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
      # Connection paths typically have a stroke but no fill
      if path_info['stroke'] and not path_info['fill']:
        connection_path_ids.add(path_id)
  for path_id, path_info in all_paths.items():
    # Identify connection paths (horizontal lines)
    if path_info['stroke_width'] == 4.0:
      connection_path_ids.add(path_id)
  return connection_path_ids


def draw_circles_with_gradients(c, root, all_gradients, ns):
  """Draw circles with gradient fills."""
  for circle in root.findall('.//svg:circle', ns):
    cx = float(circle.get('cx', '0'))
    cy = float(circle.get('cy', '0'))
    r = float(circle.get('r', '0'))
    fill = parse_color(circle.get('fill', 'none'))
    if isinstance(fill, str) and fill in all_gradients:
      grad = all_gradients[fill]
      draw_radial_gradient_shape(c, cx, cy, r, grad['stops'],
                                lambda canvas, center_x, center_y, radius: canvas.circle(center_x, center_y, radius, fill=1, stroke=0))


def draw_connection_paths(c, connection_path_ids, all_paths):
  """Draw connection paths between elements."""
  c.setLineCap(1)  # Set round cap for all connection lines
  for path_id in connection_path_ids:
    path_info = all_paths[path_id]
    # Slightly extend connection paths to ensure overlap
    commands = path_info['commands']
    draw_path(c, commands, path_info['stroke'], None, path_info['stroke_width'])


def draw_circle_shapes(c, root, all_gradients, ns):
  """Draw circle shapes."""
  for circle in root.findall('.//svg:circle', ns):
    cx = float(circle.get('cx', '0'))
    cy = float(circle.get('cy', '0'))
    r = float(circle.get('r', '0'))
    stroke = parse_color(circle.get('stroke', 'none'))
    fill = parse_color(circle.get('fill', 'none'))
    stroke_width = float(circle.get('stroke-width', '1'))
    if isinstance(fill, str) and fill in all_gradients:
      if stroke:
        c.saveState()
        c.setStrokeColorRGB(*stroke)
        c.setLineWidth(stroke_width)
        c.circle(cx, cy, r, fill=0, stroke=1)
        c.restoreState()
    else:
      draw_circle(c, cx, cy, r, stroke, fill, stroke_width)


def draw_paths(c, root, connection_path_ids, all_gradients, ns):
  """Draw regular paths that aren't connection lines."""
  for path in root.findall('.//svg:path', ns):
    parent = path.find('..')
    if parent is not None and parent.tag.endswith('defs'):
      continue  # Skip paths in defs
    path_id = path.get('id', '')
    if path_id in connection_path_ids:
      continue  # Skip connection paths, they've already been drawn
    # Skip paths with stroke-width="4.0" as these are connection lines
    if path.get('stroke-width') == '4.0':
      continue
    path_data = path.get('d', '')
    stroke = parse_color(path.get('stroke', 'none'))
    fill = parse_color(path.get('fill', 'none'))
    stroke_width = float(path.get('stroke-width', '1'))
    path_commands = parse_path(path_data)
    if isinstance(fill, str) and fill in all_gradients:
      grad = all_gradients[fill]
      first_stop = grad['stops'][0][1]  # Use color from first stop
      draw_path(c, path_commands, stroke, first_stop[:3], stroke_width)
    else:
      draw_path(c, path_commands, stroke, fill, stroke_width)


def draw_rectangles(c, root, all_gradients, ns):
  """Draw rectangle elements."""
  for rect in root.findall('.//svg:rect', ns):
    x = float(rect.get('x', '0'))
    y = float(rect.get('y', '0'))
    width = float(rect.get('width', '0'))
    height = float(rect.get('height', '0'))
    stroke = parse_color(rect.get('stroke', 'none'))
    fill = parse_color(rect.get('fill', 'none'))
    stroke_width = float(rect.get('stroke-width', '1'))
    if isinstance(fill, str) and fill in all_gradients:
      grad = all_gradients[fill]
      first_stop = grad['stops'][0][1]  # Use color from first stop
      draw_rect(c, x, y, width, height, stroke, first_stop[:3], stroke_width)
    else:
      draw_rect(c, x, y, width, height, stroke, fill, stroke_width)


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


def process_text_elements(c, root, all_paths, ns, font_to_use):
  """Process and draw text elements including text on paths."""
  for text in root.findall('.//svg:text', ns):
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
                      font_to_use, font_size, fill, text_anchor, offset_y=offset_y, is_bold=is_bold)


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
    except:
      # Continue to next filename variation if this one fails
      continue
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
    all_paths, all_gradients = extract_defs(root, ns)
    connection_path_ids = find_connection_paths(root, all_paths, ns)
    c.translate(0, height)
    c.scale(1, -1)
    c.translate(-vb_x * scale_x, -vb_y * scale_y)
    c.scale(scale_x, scale_y)
    g_element = root.find('./svg:g', ns)
    if g_element is not None:
        g_transform = g_element.get('transform', '')
        if g_transform.startswith('rotate(90'):
            pivot_match = re.search(r'rotate\(90\s+([-\d.]+)\s+([-\d.]+)\)', g_transform)
            if pivot_match:
                pivot_x, pivot_y = float(pivot_match.group(1)), float(pivot_match.group(2))
                c.translate(pivot_x, pivot_y)
                c.rotate(90)
                c.translate(-pivot_x, -pivot_y)
    draw_circles_with_gradients(c, root, all_gradients, ns)
    draw_connection_paths(c, connection_path_ids, all_paths)
    for circle_element in root.findall('.//svg:circle', ns):
        style_props_c = _parse_inline_style(circle_element.get('style', ''))
        raw_fill_c_attr = circle_element.get('fill')
        raw_fill_c = style_props_c.get('fill', raw_fill_c_attr)
        parsed_fill_c_check = parse_color(raw_fill_c if raw_fill_c is not None else ('black' if not style_props_c.get('stroke',circle_element.get('stroke')) else 'none')) # Circle default fill black
        is_gradient_fill = isinstance(parsed_fill_c_check, str) and parsed_fill_c_check in all_gradients
        if is_gradient_fill and not style_props_c.get('stroke', circle_element.get('stroke')): # If gradient and no separate stroke, skip (handled by draw_circles_with_gradients)
            continue
        raw_stroke_c = style_props_c.get('stroke', circle_element.get('stroke'))
        raw_stroke_width_c = style_props_c.get('stroke-width', circle_element.get('stroke-width','1'))
        raw_fill_opacity_c = style_props_c.get('fill-opacity', circle_element.get('fill-opacity'))
        raw_stroke_opacity_c = style_props_c.get('stroke-opacity', circle_element.get('stroke-opacity'))
        stroke_c_val = parse_color(raw_stroke_c if raw_stroke_c is not None else 'none')
        try:
            sw_c = float(str(raw_stroke_width_c).replace('px','')) if raw_stroke_width_c else 1.0
        except ValueError: sw_c = 1.0
        final_fill_c = None
        if not is_gradient_fill and parsed_fill_c_check and isinstance(parsed_fill_c_check, tuple): # Only if not gradient and fill is a color
            alpha_f_c = 1.0
            if raw_fill_opacity_c:
                try: alpha_f_c = float(raw_fill_opacity_c)
                except ValueError: pass
            final_fill_c = parsed_fill_c_check + (alpha_f_c,) if len(parsed_fill_c_check) == 3 else (parsed_fill_c_check[0],parsed_fill_c_check[1],parsed_fill_c_check[2],min(parsed_fill_c_check[3], alpha_f_c) if len(parsed_fill_c_check) == 4 else alpha_f_c)
        final_stroke_c = None
        if stroke_c_val and isinstance(stroke_c_val, tuple):
            alpha_s_c = 1.0
            if raw_stroke_opacity_c:
                try: alpha_s_c = float(raw_stroke_opacity_c)
                except ValueError: pass
            final_stroke_c = stroke_c_val + (alpha_s_c,) if len(stroke_c_val) == 3 else (stroke_c_val[0],stroke_c_val[1],stroke_c_val[2],min(stroke_c_val[3], alpha_s_c) if len(stroke_c_val) == 4 else alpha_s_c)
        cx_c = float(circle_element.get('cx', '0')); cy_c = float(circle_element.get('cy', '0')); r_c = float(circle_element.get('r', '0'))
        draw_circle(c, cx_c, cy_c, r_c, final_stroke_c, final_fill_c if not is_gradient_fill else None, sw_c)
    for rect_element in root.findall('.//svg:rect', ns):
        style_props_rect = _parse_inline_style(rect_element.get('style', ''))
        raw_fill_rect = style_props_rect.get('fill', rect_element.get('fill'))
        raw_stroke_rect = style_props_rect.get('stroke', rect_element.get('stroke'))
        raw_stroke_width_rect = style_props_rect.get('stroke-width', rect_element.get('stroke-width','1'))
        raw_fill_opacity_rect = style_props_rect.get('fill-opacity', rect_element.get('fill-opacity'))
        raw_stroke_opacity_rect = style_props_rect.get('stroke-opacity', rect_element.get('stroke-opacity'))
        raw_opacity_rect = style_props_rect.get('opacity', rect_element.get('opacity'))
        fill_r_val = parse_color(raw_fill_rect if raw_fill_rect is not None else 'none') # Rect default fill is none
        stroke_r_val = parse_color(raw_stroke_rect if raw_stroke_rect is not None else 'none')
        try:
            sw_r = float(str(raw_stroke_width_rect).replace('px','')) if raw_stroke_width_rect else 1.0
        except ValueError: sw_r = 1.0
        final_fill_r = None
        if fill_r_val and isinstance(fill_r_val, tuple):
            alpha_r = 1.0
            if raw_fill_opacity_rect:
                try: alpha_r = float(raw_fill_opacity_rect)
                except ValueError: pass
            elif raw_opacity_rect:
                try: alpha_r = float(raw_opacity_rect) # General opacity
                except ValueError: pass
            final_fill_r = fill_r_val + (alpha_r,) if len(fill_r_val) == 3 else (fill_r_val[0],fill_r_val[1],fill_r_val[2],min(fill_r_val[3],alpha_r) if len(fill_r_val)==4 else alpha_r)
        elif isinstance(fill_r_val, str) and fill_r_val in all_gradients: final_fill_r = fill_r_val # Pass ID
        final_stroke_r = None
        if stroke_r_val and isinstance(stroke_r_val, tuple):
            alpha_sr = 1.0
            if raw_stroke_opacity_rect:
                try: alpha_sr = float(raw_stroke_opacity_rect)
                except ValueError: pass
            elif raw_opacity_rect:
                try: alpha_sr = float(raw_opacity_rect)
                except ValueError: pass
            final_stroke_r = stroke_r_val + (alpha_sr,) if len(stroke_r_val) == 3 else (stroke_r_val[0],stroke_r_val[1],stroke_r_val[2],min(stroke_r_val[3],alpha_sr)if len(stroke_r_val)==4 else alpha_sr)
        x_r = float(rect_element.get('x', '0')); y_r = float(rect_element.get('y', '0'))
        w_r = float(rect_element.get('width', '0')); h_r = float(rect_element.get('height', '0'))
        if isinstance(final_fill_r, str) and final_fill_r in all_gradients:
             grad_r_data = all_gradients[final_fill_r]
             fill_color_for_rect_from_grad = None
             if grad_r_data.get('stops'): fill_color_for_rect_from_grad = grad_r_data['stops'][0][1][:3]
             draw_rect(c, x_r, y_r, w_r, h_r, final_stroke_r, fill_color_for_rect_from_grad, sw_r)
        else:
             draw_rect(c, x_r, y_r, w_r, h_r, final_stroke_r, final_fill_r, sw_r)
    for ellipse_element in root.findall('.//svg:ellipse', ns):
        style_props_ellipse = _parse_inline_style(ellipse_element.get('style', ''))
        raw_fill_e = style_props_ellipse.get('fill', ellipse_element.get('fill'))
        raw_stroke_e = style_props_ellipse.get('stroke', ellipse_element.get('stroke'))
        raw_stroke_width_e = style_props_ellipse.get('stroke-width', ellipse_element.get('stroke-width','1'))
        raw_fill_opacity_e = style_props_ellipse.get('fill-opacity', ellipse_element.get('fill-opacity'))
        raw_stroke_opacity_e = style_props_ellipse.get('stroke-opacity', ellipse_element.get('stroke-opacity'))
        fill_e_val = parse_color(raw_fill_e if raw_fill_e is not None else 'black')
        stroke_e_val = parse_color(raw_stroke_e if raw_stroke_e is not None else 'none')
        try:
            sw_e = float(str(raw_stroke_width_e).replace('px','')) if raw_stroke_width_e else 1.0
        except ValueError: sw_e = 1.0
        final_fill_e = None
        if fill_e_val and isinstance(fill_e_val, tuple):
            alpha_f_e = 1.0
            if raw_fill_opacity_e:
                try: alpha_f_e = float(raw_fill_opacity_e)
                except ValueError: pass
            final_fill_e = fill_e_val + (alpha_f_e,) if len(fill_e_val) == 3 else (fill_e_val[0],fill_e_val[1],fill_e_val[2],min(fill_e_val[3],alpha_f_e) if len(fill_e_val)==4 else alpha_f_e)
        elif isinstance(fill_e_val, str) and fill_e_val in all_gradients: final_fill_e = fill_e_val
        final_stroke_e = None
        if stroke_e_val and isinstance(stroke_e_val, tuple):
            alpha_s_e = 1.0
            if raw_stroke_opacity_e:
                try: alpha_s_e = float(raw_stroke_opacity_e)
                except ValueError: pass
            final_stroke_e = stroke_e_val + (alpha_s_e,) if len(stroke_e_val) == 3 else (stroke_e_val[0],stroke_e_val[1],stroke_e_val[2],min(stroke_e_val[3],alpha_s_e) if len(stroke_e_val)==4 else alpha_s_e)
        cx_e = float(ellipse_element.get('cx', '0')); cy_e = float(ellipse_element.get('cy', '0'))
        rx_e = float(ellipse_element.get('rx', '0')); ry_e = float(ellipse_element.get('ry', '0'))
        if isinstance(final_fill_e, str) and final_fill_e in all_gradients:
             grad_e_data = all_gradients[final_fill_e]
             fill_color_for_ellipse = None
             if grad_e_data.get('stops'): fill_color_for_ellipse = grad_e_data['stops'][0][1][:3]
             draw_ellipse(c, cx_e, cy_e, rx_e, ry_e, final_stroke_e, fill_color_for_ellipse, sw_e)
        else:
             draw_ellipse(c, cx_e, cy_e, rx_e, ry_e, final_stroke_e, final_fill_e, sw_e)
    for path_element in root.findall('.//svg:path', ns):
        is_in_defs_p = any(path_element in list(defs_el_iter) for defs_el_iter in root.findall('.//svg:defs', ns))
        if is_in_defs_p: continue
        path_id_p = path_element.get('id', '')
        if path_id_p in connection_path_ids: continue
        path_data_p = path_element.get('d', '')
        if not path_data_p: continue
        style_props_p = _parse_inline_style(path_element.get('style', ''))
        raw_fill_p = style_props_p.get('fill', path_element.get('fill'))
        raw_stroke_p = style_props_p.get('stroke', path_element.get('stroke'))
        raw_stroke_width_p = style_props_p.get('stroke-width', path_element.get('stroke-width', '1'))
        raw_fill_opacity_p = style_props_p.get('fill-opacity', path_element.get('fill-opacity'))
        raw_stroke_opacity_p = style_props_p.get('stroke-opacity', path_element.get('stroke-opacity'))
        fill_val_p = parse_color(raw_fill_p if raw_fill_p is not None else ('black' if not raw_stroke_p and not style_props_p.get('stroke') else 'none'))
        stroke_val_p = parse_color(raw_stroke_p if raw_stroke_p is not None else 'none')
        try:
            sw_p = float(str(raw_stroke_width_p).replace('px','')) if raw_stroke_width_p else 1.0
        except ValueError: sw_p = 1.0
        final_fill_p = None
        if fill_val_p and isinstance(fill_val_p, tuple):
            alpha_f = 1.0
            if raw_fill_opacity_p:
                try: alpha_f = float(raw_fill_opacity_p)
                except ValueError: pass
            final_fill_p = fill_val_p + (alpha_f,) if len(fill_val_p) == 3 else (fill_val_p[0],fill_val_p[1],fill_val_p[2],min(fill_val_p[3],alpha_f) if len(fill_val_p)==4 else alpha_f)
        elif isinstance(fill_val_p, str) and fill_val_p in all_gradients: final_fill_p = fill_val_p
        final_stroke_p = None
        if stroke_val_p and isinstance(stroke_val_p, tuple):
            alpha_s = 1.0
            if raw_stroke_opacity_p:
                try: alpha_s = float(raw_stroke_opacity_p)
                except ValueError: pass
            final_stroke_p = stroke_val_p + (alpha_s,) if len(stroke_val_p) == 3 else (stroke_val_p[0],stroke_val_p[1],stroke_val_p[2],min(stroke_val_p[3],alpha_s) if len(stroke_val_p)==4 else alpha_s)
        path_commands_p = parse_path(path_data_p)
        if isinstance(final_fill_p, str) and final_fill_p in all_gradients:
            grad_p = all_gradients[final_fill_p]
            temp_fill_color = None
            if grad_p.get('stops'):
                first_stop_p = grad_p['stops'][0][1]
                temp_fill_color = first_stop_p[:3] + (first_stop_p[3] if len(first_stop_p)>3 else 1.0,)
            draw_path(c, path_commands_p, final_stroke_p, temp_fill_color, sw_p)
        else:
            draw_path(c, path_commands_p, final_stroke_p, final_fill_p, sw_p)
    process_text_elements(c, root, all_paths, ns, font_to_use)
    return c


def convert_chem_to_file(svg_data: str, file_path: Union[str, Path, None] = None, return_bytes: bool = False):
  if isinstance(svg_data, bytes):
    svg_data = svg_data.decode('utf-8')
  ext = 'png' if file_path is None else str(file_path).lower().split('.')[-1]
  pdf_canvas_target: Union[str, Path, BytesIO]
  temp_pdf_path: Union[str, None] = None
  if ext == 'png':
    if not FITZ_AVAILABLE:
        warnings.warn("PyMuPDF (fitz) is not installed. PNG generation for 'chem' mode is not available.")
        return None
    _intermediate_temp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_pdf_path = _intermediate_temp_pdf_file.name
    _intermediate_temp_pdf_file.close()
    pdf_canvas_target = temp_pdf_path
  elif ext == 'pdf':
    if return_bytes:
      pdf_canvas_target = BytesIO()
    else:
      if file_path is None:
          raise ValueError("file_path must be provided for PDF output if not returning bytes.")
      pdf_canvas_target = file_path # type: ignore
  else:
      raise ValueError(f"Unsupported extension: {ext}")
  canvas_obj = _render_svg_to_pdf_canvas(svg_data, pdf_canvas_target, alt_text_info=None)
  canvas_obj.save()
  if ext == 'pdf':
    if return_bytes:
      pdf_data = pdf_canvas_target.getvalue() # type: ignore
      pdf_canvas_target.close() # type: ignore
      return pdf_data
    else:
      return None
  else: # ext == 'png'
    if temp_pdf_path is None:
        raise Exception("Internal error: temp PDF path for PNG (chem) not set.")
    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    if return_bytes:
      png_bytes = pix.tobytes("png")
      doc.close()
      os.unlink(temp_pdf_path)
      return png_bytes
    else: # save to file
      if file_path is None:
          doc.close()
          os.unlink(temp_pdf_path)
          raise ValueError("file_path must be provided for PNG output if not returning bytes.")
      pix.save(str(file_path))
      doc.close()
      os.unlink(temp_pdf_path)
      return None

    
def convert_svg_to_pdf(svg_data: str, pdf_file_path: Union[str, Path], return_canvas: bool = False, chem: bool = False):
  if chem:
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    convert_chem_to_file(svg_data, file_path=pdf_file_path, return_bytes=False)
    return None
  if isinstance(svg_data, bytes):
    svg_data = svg_data.decode('utf-8')
  alt_text_payload = None
  aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
  if aria_label_match:
    alt_text_payload = {'alt_text': aria_label_match.group(1)}
  canvas_obj = _render_svg_to_pdf_canvas(svg_data, pdf_file_path, alt_text_info=alt_text_payload)
  if return_canvas:
    return canvas_obj
  else:
    canvas_obj.save()
    return None


def convert_svg_to_png(svg_data: str, png_file_path: Union[str, Path, None] = None,
                       output_width: Union[int, None] = None, output_height: Union[int, None] = None,
                       scale: Union[float, None] = None, return_bytes: bool = False,
                       chem: bool = False):
  if chem:
    if isinstance(svg_data, bytes):
        svg_data = svg_data.decode('utf-8')
    return convert_chem_to_file(svg_data, file_path=png_file_path, return_bytes=return_bytes)
  if not FITZ_AVAILABLE:
    warnings.warn(
        "PyMuPDF (fitz) is not installed. PNG generation is not available. "
        "Please install PyMuPDF (`pip install PyMuPDF`) to enable PNG export."
    )
    if return_bytes: return None
    else: return None
  if not return_bytes and png_file_path is None:
      raise ValueError("png_file_path must be provided if return_bytes is False.")
  if isinstance(svg_data, bytes):
    svg_data = svg_data.decode('utf-8')
  aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
  alt_text = aria_label_match.group(1) if aria_label_match else None
  alt_text_payload_for_pdf = {'alt_text': alt_text} if alt_text else None
  temp_pdf_path_local: str
  if png_file_path is None and return_bytes is True:
    _temp_pdf_obj_bytes = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_pdf_path_local = _temp_pdf_obj_bytes.name
    _temp_pdf_obj_bytes.close()
  elif png_file_path is not None:
    _temp_pdf_obj_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_pdf_path_local = _temp_pdf_obj_file.name
    _temp_pdf_obj_file.close()
  else: # Should not happen given the initial check
    raise ValueError("Invalid state for png_file_path and return_bytes")
  canvas_object = _render_svg_to_pdf_canvas(svg_data, temp_pdf_path_local, alt_text_info=alt_text_payload_for_pdf)
  canvas_object.save()
  doc = fitz.open(temp_pdf_path_local)
  page = doc[0]
  page_rect = page.rect
  page_width = page_rect.width if page_rect.width > 1e-3 else 1.0
  page_height = page_rect.height if page_rect.height > 1e-3 else 1.0
  zoom_x, zoom_y = 1.0, 1.0
  if scale is not None:
      zoom_x = zoom_y = scale
  elif output_width is not None and output_height is not None:
      zoom_x = output_width / page_width
      zoom_y = output_height / page_height
  elif output_width is not None:
      zoom_x = zoom_y = output_width / page_width
  elif output_height is not None:
      zoom_x = zoom_y = output_height / page_height
  zoom_matrix = fitz.Matrix(zoom_x, zoom_y)
  pix = page.get_pixmap(matrix=zoom_matrix)
  if return_bytes:
    png_data_bytes = pix.tobytes("png")
    doc.close()
    os.unlink(temp_pdf_path_local)
    return png_data_bytes
  else: # Save to file
    if png_file_path is None: # Already checked, but for safety
        doc.close()
        os.unlink(temp_pdf_path_local)
        raise ValueError("png_file_path is required for PNG file output.")
    pix.save(str(png_file_path))
    doc.close()
    os.unlink(temp_pdf_path_local)
    if alt_text and png_file_path:
        try:
            img = Image.open(str(png_file_path))
            img.load()
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("alt", alt_text)
            img.save(str(png_file_path), pnginfo=metadata)
        except Exception:
            pass
    return None
