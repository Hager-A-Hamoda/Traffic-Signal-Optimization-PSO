from .traffic_light import TrafficLight
from .intersection import Intersection
from .road import Road
from .traffic_network import TrafficNetwork

def build_network():
    # Traffic Lights (green, red)
    tl1 = TrafficLight(10, 10)
    tl2 = TrafficLight(15, 10)
    tl3 = TrafficLight(12, 10)

    # Intersections
    i1 = Intersection(tl1, lanes=2)
    i2 = Intersection(tl2, lanes=1)
    i3 = Intersection(tl3, lanes=2)

    # Roads
    r1 = Road(delay=3)
    r2 = Road(delay=2)

    # Connections
    i1.add_outgoing(r1)
    i2.add_incoming(r1)

    i2.add_outgoing(r2)
    i3.add_incoming(r2)

    return TrafficNetwork([i1, i2, i3])