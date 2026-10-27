"""Shared UDP packet format: type (1 B) + sequence number (4 B) + send time (8 B)."""
import struct

HEADER = struct.Struct("!BId")
TYPE_DATA = 0
TYPE_END = 1


def pack_packet(ptype: int, seq: int, send_time: float, payload: bytes = b"") -> bytes:
    return HEADER.pack(ptype, seq, send_time) + payload


def unpack_header(data: bytes):
    """Return (type, seq, send_time). Raises ValueError if the packet is too short."""
    if len(data) < HEADER.size:
        raise ValueError("packet shorter than header")
    return HEADER.unpack_from(data)