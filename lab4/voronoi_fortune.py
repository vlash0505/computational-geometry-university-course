import math
import itertools
import heapq


class Point:
    """
    Represents a point in 2D space.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
    """
    def __init__(self, x, y):
        """
        Initializes a Point instance.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
        """
        self.x = x
        self.y = y


class Event:
    """
    Represents an event in the Voronoi diagram algorithm.

    Attributes:
        x (float): The x-coordinate of the event.
        p (Point): The point associated with the event.
        a (Arc): The arc associated with the event.
        valid (bool): Indicates if the event is valid.
    """
    def __init__(self, x, p, a):
        """
        Initializes an Event instance.

        Args:
            x (float): The x-coordinate of the event.
            p (Point): The point associated with the event.
            a (Arc): The arc associated with the event.
        """
        self.x = x
        self.p = p
        self.a = a
        self.valid = True


class Arc:
    """
    Represents an arc in the Voronoi diagram.

    Attributes:
        p (Point): The focus point of the arc.
        pprev (Arc): The previous arc in the beachline sequence.
        pnext (Arc): The next arc in the beachline sequence.
        e (Event): The circle event associated with the arc.
        s0 (Segment): The left segment of the edge of the Voronoi cell.
        s1 (Segment): The right segment of the edge of the Voronoi cell.
    """
    def __init__(self, p, a=None, b=None):
        """
        Initializes an Arc instance.

        Args:
            p (Point): The focus point of the arc.
            a (Arc, optional): The previous arc in the beachline sequence.
            b (Arc, optional): The next arc in the beachline sequence.
        """
        self.p = p
        self.pprev = a
        self.pnext = b
        self.e = None
        self.s0 = None
        self.s1 = None


class Segment:
    """
    Represents a segment in the Voronoi diagram.

    Attributes:
        start (Point): The start point of the segment.
        end (Point): The end point of the segment.
        done (bool): Indicates if the segment is complete.
    """
    def __init__(self, p):
        """
        Initializes a Segment instance.

        Args:
            p (Point): The start point of the segment.
        """
        self.start = p
        self.end = None
        self.done = False

    def finish(self, p):
        """
        Finishes the segment by setting the end point.

        Args:
            p (Point): The end point of the segment.
        """
        if not self.done:
            self.end = p
            self.done = True


class EventPriorityQueue:
    """
    Represents a priority queue for Events.

    Attributes:
        pq (list): The underlying data structure for the queue.
        entry_finder (dict): Maps tasks to entries.
        counter (itertools.count): A counter for the number of tasks in the queue.
    """
    def __init__(self):
        """
        Initializes an EventPriorityQueue instance.
        """
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, item):
        """
        Adds an item to the queue.

        Args:
            item (Event): The item to add.
        """
        if item not in self.entry_finder:
            count = next(self.counter)
            entry = [item.x, count, item]  # priority, index, item
            self.entry_finder[item] = entry
            heapq.heappush(self.pq, entry)

    def remove_entry(self, item):
        """
        Removes an item from the queue.

        Args:
            item (Event): The item to remove.
        """
        entry = self.entry_finder.pop(item)
        entry[-1] = 'Removed'  # Mark as removed

    def pop(self):
        """
        Removes and returns the lowest priority task.

        Returns:
            Event: The lowest priority event.

        Raises:
            KeyError: If the priority queue is empty.
        """
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def top(self):
        """
        Returns the lowest priority task without removing it.

        Returns:
            Event: The lowest priority event.

        Raises:
            KeyError: If the priority queue is empty.
        """
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                self.push(item)  # Push the item back as it is not being removed
                return item
        raise KeyError('top from an empty priority queue')

    def empty(self):
        """
        Checks if the priority queue is empty.

        Returns:
            bool: True if the priority queue is empty, False otherwise.
        """
        return not self.pq


class Voronoi:
    def __init__(self, points):
        """
        Initializes the Voronoi class with the given points.

        Args:
            points (list): List of points (x, y) for creating Voronoi diagram.
        """
        self.output = []  # list of line segments
        self.arc = None  # binary tree for parabola arcs

        self.points = EventPriorityQueue()  # site events
        self.event = EventPriorityQueue()  # circle events

        # bounding box
        self.x0 = -50.0
        self.x1 = -50.0
        self.y0 = 550.0
        self.y1 = 550.0

        # insert points to site event
        for pts in points:
            point = Point(pts[0], pts[1])
            self.points.push(point)
            # keep track of bounding box size
            if point.x < self.x0:
                self.x0 = point.x
            if point.y < self.y0:
                self.y0 = point.y
            if point.x > self.x1:
                self.x1 = point.x
            if point.y > self.y1:
                self.y1 = point.y

        # add margins to the bounding box
        dx = (self.x1 - self.x0 + 1) / 5.0
        dy = (self.y1 - self.y0 + 1) / 5.0
        self.x0 = self.x0 - dx
        self.x1 = self.x1 + dx
        self.y0 = self.y0 - dy
        self.y1 = self.y1 + dy

    def process(self):
        """
        Process the points and generate the Voronoi diagram.
        """
        while not self.points.empty():
            if not self.event.empty() and (self.event.top().x <= self.points.top().x):
                self.process_event()  # handle circle event
            else:
                self.process_point()  # handle site event

        # after all points, process remaining circle events
        while not self.event.empty():
            self.process_event()

        self.finish_edges()

    def process_point(self):
        """
        Process the next site event point.
        """
        # get next event from site pq
        p = self.points.pop()
        # add new arc (parabola)
        self.arc_insert(p)

    def process_event(self):
        """
        Process the next circle event.
        """
        # get next event from circle pq
        e = self.event.pop()

        if e.valid:
            # start new edge
            s = Segment(e.p)
            self.output.append(s)

            # remove associated arc (parabola)
            a = e.a
            if a.pprev is not None:
                a.pprev.pnext = a.pnext
                a.pprev.s1 = s
            if a.pnext is not None:
                a.pnext.pprev = a.pprev
                a.pnext.s0 = s

            # finish the edges before and after a
            if a.s0 is not None:
                a.s0.finish(e.p)
            if a.s1 is not None:
                a.s1.finish(e.p)

            # recheck circle events on either side of p
            if a.pprev is not None:
                self.check_circle_event(a.pprev, e.x)
            if a.pnext is not None:
                self.check_circle_event(a.pnext, e.x)

    def arc_insert(self, p):
        """
        Insert a new arc (parabola) into the beachline.

        Args:
            p (Point): The point at which the new arc is inserted.
        """
        if self.arc is None:
            self.arc = Arc(p)
        else:
            # find the current arcs at p.y
            i = self.arc
            while i is not None:
                flag, z = self.intersect(p, i)
                if flag:
                    # new parabola intersects arc i
                    flag, zz = self.intersect(p, i.pnext)
                    if (i.pnext is not None) and (not flag):
                        i.pnext.pprev = Arc(i.p, i, i.pnext)
                        i.pnext = i.pnext.pprev
                    else:
                        i.pnext = Arc(i.p, i)
                    i.pnext.s1 = i.s1

                    # add p between i and i.pnext
                    i.pnext.pprev = Arc(p, i, i.pnext)
                    i.pnext = i.pnext.pprev

                    i = i.pnext  # now i points to the new arc

                    # add new half-edges connected to i's endpoints
                    seg = Segment(z)
                    self.output.append(seg)
                    i.pprev.s1 = i.s0 = seg

                    seg = Segment(z)
                    self.output.append(seg)
                    i.pnext.s0 = i.s1 = seg

                    # check for new circle events around the new arc
                    self.check_circle_event(i, p.x)
                    self.check_circle_event(i.pprev, p.x)
                    self.check_circle_event(i.pnext, p.x)

                    return

                i = i.pnext

            # if p never intersects an arc, append it to the list
            i = self.arc
            while i.pnext is not None:
                i = i.pnext
            i.pnext = Arc(p, i)

            # insert new segment between p and i
            x = self.x0
            y = (i.pnext.p.y + i.p.y) / 2.0
            start = Point(x, y)

            seg = Segment(start)
            i.s1 = i.pnext.s0 = seg
            self.output.append(seg)

    def check_circle_event(self, i, x0):
        """
        Check for a new circle event for the given arc.

        Args:
            i (Arc): The arc to check for circle event.
            x0 (float): The x-coordinate of the sweepline.
        """
        # look for a new circle event for arc i
        if (i.e is not None) and (i.e.x != self.x0):
            i.e.valid = False
        i.e = None

        if (i.pprev is None) or (i.pnext is None):
            return

        flag, x, o = self.circle(i.pprev.p, i.p, i.pnext.p)
        if flag and (x > self.x0):
            i.e = Event(x, o, i)
            self.event.push(i.e)

    def circle(self, a, b, c):
        """
        Check if the points a, b, and c are in a counterclockwise order and calculate the circle parameters.

        Args:
            a (Point): First point.
            b (Point): Second point.
            c (Point): Third point.

        Returns:
            tuple: A tuple containing three values:
                   - flag (bool): Indicates whether points are in a counterclockwise order.
                   - x (float): The x-coordinate of the center of the circle.
                   - o (Point): The center of the circle.
        """
        # check if bc is a "right turn" from ab
        if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) > 0:
            return False, None, None

        A = b.x - a.x
        B = b.y - a.y
        C = c.x - a.x
        D = c.y - a.y
        E = A * (a.x + b.x) + B * (a.y + b.y)
        F = C * (a.x + c.x) + D * (a.y + c.y)
        G = 2 * (A * (c.y - b.y) - B * (c.x - b.x))

        if G == 0:
            return False, None, None  # Points are co-linear

        # point o is the center of the circle
        ox = 1.0 * (D * E - B * F) / G
        oy = 1.0 * (A * F - C * E) / G

        # o.x plus radius equals max x coord
        x = ox + math.sqrt((a.x - ox) ** 2 + (a.y - oy) ** 2)
        o = Point(ox, oy)

        return True, x, o

    def intersect(self, p, i):
        """
        Check whether a new parabola at point p intersects with arc i.

        Args:
            p (Point): The point of the new parabola.
            i (Arc): The arc to check intersection with.

        Returns:
            tuple: A tuple containing two values:
                   - flag (bool): Indicates whether the parabola intersects the arc.
                   - res (Point): The intersection point.
        """
        if i is None:
            return False, None
        if i.p.x == p.x:
            return False, None

        a = 0.0
        b = 0.0

        if i.pprev is not None:
            a = (self.intersection(i.pprev.p, i.p, 1.0 * p.x)).y
        if i.pnext is not None:
            b = (self.intersection(i.p, i.pnext.p, 1.0 * p.x)).y

        if (((i.pprev is None) or (a <= p.y)) and ((i.pnext is None) or (p.y <= b))):
            py = p.y
            px = 1.0 * ((i.p.x) ** 2 + (i.p.y - py) ** 2 - p.x ** 2) / (2 * i.p.x - 2 * p.x)
            res = Point(px, py)
            return True, res
        return False, None

    def intersection(self, p0, p1, l):
        """
        Get the intersection of two parabolas.

        Args:
            p0 (Point): The first focus point of the parabola.
            p1 (Point): The second focus point of the parabola.
            l (float): The x-coordinate of the intersection point.

        Returns:
            Point: The intersection point of the two parabolas.
        """
        # get the intersection of two parabolas
        p = p0
        if p0.x == p1.x:
            py = (p0.y + p1.y) / 2.0
        elif p1.x == l:
            py = p1.y
        elif p0.x == l:
            py = p0.y
            p = p1
        else:
            # use quadratic formula
            z0 = 2.0 * (p0.x - l)
            z1 = 2.0 * (p1.x - l)

            a = 1.0 / z0 - 1.0 / z1
            b = -2.0 * (p0.y / z0 - p1.y / z1)
            c = 1.0 * (p0.y ** 2 + p0.x ** 2 - l ** 2) / z0 - 1.0 * (p1.y ** 2 + p1.x ** 2 - l ** 2) / z1

            py = 1.0 * (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)

        px = 1.0 * (p.x ** 2 + (p.y - py) ** 2 - l ** 2) / (2 * p.x - 2 * l)
        res = Point(px, py)
        return res

    def finish_edges(self):
        """
        Finish the edges of the Voronoi diagram.
        """
        l = self.x1 + (self.x1 - self.x0) + (self.y1 - self.y0)
        i = self.arc
        while i.pnext is not None:
            if i.s1 is not None:
                p = self.intersection(i.p, i.pnext.p, l * 2.0)
                i.s1.finish(p)
            i = i.pnext

    def print_output(self):
        """
        Print the output of the Voronoi diagram.
        """
        it = 0
        for o in self.output:
            it = it + 1
            p0 = o.start
            p1 = o.end
            print(p0.x, p0.y, p1.x, p1.y)

    def get_output(self):
        """
        Get the output of the Voronoi diagram.

        Returns:
            list: List of tuples representing line segments in the Voronoi diagram.
        """
        res = []
        for o in self.output:
            p0 = o.start
            p1 = o.end
            res.append((p0.x, p0.y, p1.x, p1.y))
        return res
