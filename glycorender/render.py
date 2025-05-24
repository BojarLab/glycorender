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
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
import fitz
from svglib.svglib import svg2rlg
from io import BytesIO
from PIL import Image, PngImagePlugin

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
  if len(commands) == 5:  # M + 3 L + Z potentially
    cmd_types = [cmd for cmd, _ in commands]
    if cmd_types[0] in ['M', 'm'] and cmd_types[-1] in ['Z', 'z'] and all(c in ['L', 'l'] for c in cmd_types[1:-1]):
      # Extract points for possible diamond
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
  # Check if this is a bracket part
  is_bracket = False
  if len(commands) == 2:
    cmd1, params1 = commands[0]
    cmd2, params2 = commands[1]
    if cmd1 == 'M' and cmd2 in ['L', 'V', 'H']:
      # Vertical line in bracket
      if cmd2 == 'V' or (cmd2 == 'L' and len(params2) >= 2 and abs(params1[0] - params2[0]) < 0.1):
        is_bracket = True
      # Horizontal line in bracket
      elif cmd2 == 'H' or (cmd2 == 'L' and len(params2) >= 2 and abs(params1[1] - params2[1]) < 0.1 and abs(params1[0] - params2[0]) < 15):
        is_bracket = True
  if is_bracket:
    # For bracket components
    c.setLineJoin(0)  # Mitered joins
    c.setLineCap(0)   # Butt caps
  else:
    # For diamonds, connections, and other shapes
    c.setLineJoin(1)  # Round joins
    # For connection paths with stroke_width=4.0, use round caps
    if stroke_width == 4.0 and not fill_color:
      c.setLineCap(1)  # Round caps
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
    curr_x, curr_y = 0, 0
    start_x, start_y = None, None
    first_x, first_y = None, None
    for cmd, params in commands:
      if cmd == 'M':
        for i in range(0, len(params), 2):
          curr_x, curr_y = params[i], params[i+1]
          if first_x is None:
            first_x, first_y = curr_x, curr_y
          if start_x is None:
            start_x, start_y = curr_x, curr_y
          path.moveTo(curr_x, curr_y)
      elif cmd == 'm':
        for i in range(0, len(params), 2):
          curr_x += params[i]
          curr_y += params[i+1]
          if first_x is None:
            first_x, first_y = curr_x, curr_y
          if start_x is None:
            start_x, start_y = curr_x, curr_y
          path.moveTo(curr_x, curr_y)
      elif cmd == 'L':
        for i in range(0, len(params), 2):
          curr_x, curr_y = params[i], params[i+1]
          path.lineTo(curr_x, curr_y)
      elif cmd == 'l':
        for i in range(0, len(params), 2):
          curr_x += params[i]
          curr_y += params[i+1]
          path.lineTo(curr_x, curr_y)
      elif cmd == 'H':
        for param in params:
          curr_x = param
          path.lineTo(curr_x, curr_y)
      elif cmd == 'h':
        for param in params:
          curr_x += param
          path.lineTo(curr_x, curr_y)
      elif cmd == 'V':
        for param in params:
          curr_y = param
          path.lineTo(curr_x, curr_y)
      elif cmd == 'v':
        for param in params:
          curr_y += param
          path.lineTo(curr_x, curr_y)
      elif cmd == 'Z' or cmd == 'z':
        if first_x is not None and first_y is not None:
          path.lineTo(first_x, first_y)
        path.close()
    if first_x is not None and (cmd != 'Z' and cmd != 'z') and (curr_x != first_x or curr_y != first_y):
      path.lineTo(first_x, first_y)
      path.close()
  # Draw path with fill if fill color is provided, regardless of stroke width
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


def extract_defs(root, ns):
  """Extract definitions like paths and gradients from SVG."""
  all_paths = {}
  all_gradients = {}
  for defs in root.findall('.//svg:defs', ns):
    # Extract paths
    for path in defs.findall('.//svg:path', ns):
      path_id = path.get('id', '')
      if not path_id:
        continue
      path_data = path.get('d', '')
      stroke = parse_color(path.get('stroke', 'none'))
      fill = parse_color(path.get('fill', 'none'))
      stroke_width = float(path.get('stroke-width', '1'))
      path_commands = parse_path(path_data)
      path_points = calculate_points(path_commands)
      all_paths[path_id] = {
        'points': path_points,
        'commands': path_commands,
        'stroke': stroke,
        'fill': fill,
        'stroke_width': stroke_width
      }
    # Extract radial gradients
    for radial_gradient in defs.findall('.//svg:radialGradient', ns):
      gradient_id = radial_gradient.get('id', '')
      if not gradient_id:
        continue
      cx = float(radial_gradient.get('cx', '0'))
      cy = float(radial_gradient.get('cy', '0'))
      r = float(radial_gradient.get('r', '0'))
      stops = []
      for stop in radial_gradient.findall('.//svg:stop', ns):
        offset = float(stop.get('offset', '0'))
        stop_color = stop.get('stop-color', 'white')
        opacity = float(stop.get('stop-opacity', '1'))
        color_tuple = None
        if stop_color.startswith('#'):
          if len(stop_color) == 7:  # #RRGGBB
            r_val = int(stop_color[1:3], 16) / 255.0
            g_val = int(stop_color[3:5], 16) / 255.0
            b_val = int(stop_color[5:7], 16) / 255.0
            color_tuple = (r_val, g_val, b_val, opacity)
        elif stop_color in reportlab_colors:
          base_color = reportlab_colors[stop_color]
          color_tuple = base_color + (opacity,)
        if color_tuple:
          stops.append((offset, color_tuple))
      all_gradients[gradient_id] = {'cx': cx, 'cy': cy, 'r': r,'stops': stops}
  # Also process direct paths with stroke-width="4.0" not in defs
  for path in root.findall('.//svg:path', ns):
    if path.get('stroke-width') == '4.0':
      parent = path.find('..')
      if parent is not None and parent.tag.endswith('defs'):
        continue  # Skip if in defs (already processed)
      path_id = f"connection_{len(all_paths)}"  # Generate unique ID
      path_data = path.get('d', '')
      stroke = parse_color(path.get('stroke', 'none'))
      fill = parse_color(path.get('fill', 'none'))
      stroke_width = float(path.get('stroke-width', '1'))
      path_commands = parse_path(path_data)
      path_points = calculate_points(path_commands)
      all_paths[path_id] = {
        'points': path_points,
        'commands': path_commands,
        'stroke': stroke,
        'fill': fill,
        'stroke_width': stroke_width
      }
  return all_paths, all_gradients


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


def convert_chem_to_file(svg_data, file_path=None, return_bytes=False):
  """
  Convert a chemical 2D depiction from RDKit into a .pdf/.png
  If return_bytes is True, returns the file contents as bytes instead of saving to disk.
  """
  width_match = re.search(r'width=[\'"](\d+)px[\'"]', svg_data)
  height_match = re.search(r'height=[\'"](\d+)px[\'"]', svg_data)
  width = int(width_match.group(1)) if width_match else 250
  height = int(height_match.group(1)) if height_match else 250
  svg_bytes = svg_data.encode('utf-8')
  drawing = svg2rlg(BytesIO(svg_bytes))
  drawing.width, drawing.height = width, height
  # Determine output format based on file extension or default
  ext = 'png' if file_path is None else file_path.lower().split('.')[-1]
  # For PNG output
  if ext == 'png':
    # Create a temporary PDF
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_pdf_path = temp_pdf.name
    temp_pdf.close()
    # Create PDF
    renderPDF.drawToFile(drawing, temp_pdf_path)
    # Convert PDF to PNG using fitz
    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)  # Adjust DPI for quality
    if return_bytes:
      png_bytes = pix.tobytes("png")
      doc.close()
      os.unlink(temp_pdf_path)
      return png_bytes
    else:
      pix.save(file_path)
      doc.close()
      os.unlink(temp_pdf_path)
      return None
  # For PDF output
  else:
    if return_bytes:
      # Create a temporary file for PDF
      temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
      temp_pdf_path = temp_pdf.name
      temp_pdf.close()
      # Generate the PDF
      renderPDF.drawToFile(drawing, temp_pdf_path)
      # Read the file contents
      with open(temp_pdf_path, 'rb') as f:
        pdf_bytes = f.read()
      # Clean up
      os.unlink(temp_pdf_path)
      return pdf_bytes
    else:
      renderPDF.drawToFile(drawing, file_path)
      return None


def convert_svg_to_pdf(svg_data, pdf_file_path, return_canvas=False, chem=False):
  """Convert SVG to PDF with text path support."""
  if isinstance(svg_data, bytes):
    svg_data = svg_data.decode('utf-8')
  if chem:
    convert_chem_to_file(svg_data, pdf_file_path)
    return None
  # Extract ALT text from SVG
  aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
  alt_text = aria_label_match.group(1) if aria_label_match else None
  root = ET.fromstring(svg_data)
  ns = {'svg': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'}
  # Parse dimensions and set up canvas
  width, height, vb_x, vb_y, scale_x, scale_y = parse_svg_dimensions(root)
  c = canvas.Canvas(pdf_file_path, pagesize=(width, height))
  if alt_text:
    c.setTitle(alt_text.replace("SNFG diagram of ", "").split(" drawn in")[0])
    c.setAuthor("GlycoDraw")
    c.setSubject("Glycan Visualization")
    c.setKeywords(f"glycan;carbohydrate;glycowork;Description: {alt_text}")
  # Extract definitions and find connection paths
  all_paths, all_gradients = extract_defs(root, ns)
  connection_path_ids = find_connection_paths(root, all_paths, ns)
  # Set up canvas transformation
  c.translate(0, height)
  c.scale(1, -1)
  c.translate(-vb_x * scale_x, -vb_y * scale_y)
  c.scale(scale_x, scale_y)
  g_transform = root.find('.//{http://www.w3.org/2000/svg}g').get('transform', '')
  if g_transform.startswith('rotate(90'):
    pivot_match = re.search(r'rotate\(90\s+([-\d.]+)\s+([-\d.]+)', g_transform)
    if pivot_match:
      pivot_x, pivot_y = float(pivot_match.group(1)), float(pivot_match.group(2))
      c.translate(pivot_x, pivot_y)
      c.rotate(90)
      c.translate(-pivot_x, -pivot_y)
  # Draw elements in the correct order
  draw_circles_with_gradients(c, root, all_gradients, ns)
  draw_connection_paths(c, connection_path_ids, all_paths)
  draw_circle_shapes(c, root, all_gradients, ns)
  draw_rectangles(c, root, all_gradients, ns)
  draw_paths(c, root, connection_path_ids, all_gradients, ns)
  process_text_elements(c, root, all_paths, ns, font_to_use)
  if return_canvas:
    return c
  c.save()


def convert_svg_to_png(svg_data, png_file_path=None, output_width=None, output_height=None, scale=None, return_bytes=False,
                       chem=False):
  """Convert SVG to PNG with pymupdf, with support for scaling and dimensions."""
  if chem:
    if return_bytes:
      return convert_chem_to_file(svg_data, png_file_path, return_bytes=True)
    convert_chem_to_file(svg_data, png_file_path)
    return None
  # Extract ALT text from SVG
  aria_label_match = re.search(r'aria-label=["\']([^"\']+)["\']', svg_data)
  alt_text = aria_label_match.group(1) if aria_label_match else None
  temp_pdf = None
  if png_file_path is None:
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
      temp_pdf = tmp.name
      pdf_path = temp_pdf
  else:
    pdf_path = f"{png_file_path.split('.')[0]}.pdf"
  canvas_object = convert_svg_to_pdf(svg_data, pdf_path, return_canvas=True)
  canvas_object.save()
  doc = fitz.open(canvas_object._filename)
  page = doc[0]
  zoom_matrix = fitz.Matrix(1, 1)
  if scale is not None:
    zoom_matrix = fitz.Matrix(scale, scale)
  elif output_width is not None or output_height is not None:
    page_rect = page.rect
    page_width, page_height = page_rect.width, page_rect.height
    scale_x = output_width / page_width if output_width else 1
    scale_y = output_height / page_height if output_height else 1
    zoom_matrix = fitz.Matrix(scale_x, scale_y)
  pix = page.get_pixmap(matrix=zoom_matrix)
  if return_bytes:
    png_data = pix.tobytes("png")
    doc.close()
    if temp_pdf:
      os.remove(temp_pdf)
    return png_data
  else:
    pix.save(png_file_path)
    doc.close()
    os.remove(pdf_path)
    # Add ALT text metadata to PNG using Pillow
    if alt_text and png_file_path:
      img = Image.open(png_file_path)
      metadata = PngImagePlugin.PngInfo()
      metadata.add_text("alt", alt_text)
      img.save(png_file_path, pnginfo=metadata)
    return None
