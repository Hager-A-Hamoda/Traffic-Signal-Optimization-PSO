from .traffic_light import TrafficLight
from .intersection import Intersection
from .road import Road
from .traffic_network import TrafficNetwork

def build_network():
    # Traffic lights
    tl1 = TrafficLight(10, 8)
    tl2 = TrafficLight(12, 8)
    tl3 = TrafficLight(10, 10)

    # Intersections
    i1 = Intersection(tl1, lanes=2)
    i2 = Intersection(tl2, lanes=1)
    i3 = Intersection(tl3, lanes=2)

    # Main Roads
    r1 = Road(delay=2)
    r2 = Road(delay=2)

    # Side Roads
    s1_in = Road(delay=3)
    s1_out = Road(delay=3)

    s2_in = Road(delay=3)
    s2_out = Road(delay=3)

    s3_in = Road(delay=3)
    s3_out = Road(delay=3)

    # Connections 
    # I1
    i1.add_side_incoming(s1_in)
    i1.add_main_outgoing(r1)
    i1.add_side_outgoing(s1_out)

    # I2
    i2.add_main_incoming(r1)
    i2.add_side_incoming(s2_in)
    i2.add_main_outgoing(r2)
    i2.add_side_outgoing(s2_out)

    # I3
    i3.add_main_incoming(r2)
    i3.add_side_incoming(s3_in)
    i3.add_side_outgoing(s3_out)

    return TrafficNetwork([i1, i2, i3])