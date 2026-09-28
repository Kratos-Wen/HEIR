import torch
from PIL import Image

from .tokenizer import letterbox, transform_boxes


def test_nonsquare_box_transform_roundtrip():
    for size in ((641, 300), (301, 640), (512, 512)):
        tensor, geometry = letterbox(Image.new('RGB', size))
        boxes = torch.tensor([[0., 0., size[0], size[1]], [10., 20., 100., 200.]])
        mapped = transform_boxes(boxes, geometry)
        torch.testing.assert_close(transform_boxes(mapped, geometry, inverse=True), boxes, atol=1e-4, rtol=1e-5)
        assert tensor.shape == (3, 512, 512)
        assert mapped.min() >= 0 and mapped.max() <= 512
