"""Gradient compression codec: float16 quantization + lz4 block compression.

Wire format (compressed):
  [magic: 1 byte = 0x01]
  [original_size: uint32_le]      # decompressed float16 payload size
  [lz4_block_compressed(float16_binary_payload)]

If the first byte is NOT 0x01, the payload is treated as legacy float32 binary
(backward compatible).

The float16 binary layout mirrors the float32 format from fed_avg.py:
  [layer_count: uint32_le]
  For each layer:
    [name_length: uint32_le]
    [name: utf8_bytes]
    [element_count: uint32_le]
    [values: float16_le × element_count]   # 2 bytes per element
"""

import struct

import lz4.block
import numpy as np

MAGIC = 0x01


def compress_gradients(raw_float32_binary: bytes) -> bytes:
    """Quantize float32 binary to float16, lz4-compress, and wrap with header."""
    f16_payload = _quantize_f32_to_f16(raw_float32_binary)
    compressed = lz4.block.compress(f16_payload, store_size=False)
    header = struct.pack("<BI", MAGIC, len(f16_payload))
    return header + compressed


def decompress_gradients(data: bytes) -> bytes:
    """Detect magic byte; decompress lz4 + dequantize float16→float32, or passthrough."""
    if len(data) < 1 or data[0] != MAGIC:
        return data  # legacy float32 passthrough

    original_size = struct.unpack_from("<I", data, 1)[0]
    compressed_payload = data[5:]
    f16_payload = lz4.block.decompress(compressed_payload, uncompressed_size=original_size)
    return _dequantize_f16_to_f32(f16_payload)


def _quantize_f32_to_f16(data: bytes) -> bytes:
    """Convert float32 values to float16 in the binary gradient format."""
    offset = 0
    (layer_count,) = struct.unpack_from("<I", data, offset)
    offset += 4

    parts: list[bytes] = [struct.pack("<I", layer_count)]

    for _ in range(layer_count):
        (name_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name = data[offset : offset + name_len]
        offset += name_len
        (elem_count,) = struct.unpack_from("<I", data, offset)
        offset += 4

        values_f32 = np.frombuffer(data, dtype=np.float32, count=elem_count, offset=offset)
        offset += elem_count * 4
        values_f16 = values_f32.astype(np.float16)

        parts.append(struct.pack("<I", name_len))
        parts.append(name)
        parts.append(struct.pack("<I", elem_count))
        parts.append(values_f16.tobytes())

    return b"".join(parts)


def _dequantize_f16_to_f32(data: bytes) -> bytes:
    """Convert float16 values back to float32 in the binary gradient format."""
    offset = 0
    (layer_count,) = struct.unpack_from("<I", data, offset)
    offset += 4

    parts: list[bytes] = [struct.pack("<I", layer_count)]

    for _ in range(layer_count):
        (name_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name = data[offset : offset + name_len]
        offset += name_len
        (elem_count,) = struct.unpack_from("<I", data, offset)
        offset += 4

        values_f16 = np.frombuffer(data, dtype=np.float16, count=elem_count, offset=offset)
        offset += elem_count * 2
        values_f32 = values_f16.astype(np.float32)

        parts.append(struct.pack("<I", name_len))
        parts.append(name)
        parts.append(struct.pack("<I", elem_count))
        parts.append(values_f32.tobytes())

    return b"".join(parts)
