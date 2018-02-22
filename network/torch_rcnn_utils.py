import math
import torch

_LOG_2 = math.log(2.0)


def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), _LOG_2)


def torch_apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, :, 0] * height
    center_x += deltas[:, :, 1] * width
    height *= torch.exp(deltas[:, :, 2])
    width *= torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = [y1, x1, y2, x2]
    return result


def torch_clip_boxes(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = window
    y1, x1, y2, x2 = boxes
    # Clip

    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)

    clipped = torch.stack([x1, y1, x2, y2], dim=2)
    return clipped

