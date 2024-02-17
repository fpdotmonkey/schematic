use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use godot::builtin::{
    meta::{ConvertError, FromGodot, GodotConvert, ToGodot},
    Vector2i,
};
use slotmap::Key;

/// The computational side of a circuit diagram
///
/// This handles working out how to route all the lines and junctions in
/// a pleasing way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedding {
    connections: Vec<(JunctionKey, JunctionKey)>,
    inner: EmbeddingInner,
}

/// An anchor for a connection
#[derive(Debug, Clone, PartialEq)]
pub struct Junction {
    position: Vector2i,
    voltage: u32,
    kind: (),
}

slotmap::new_key_type! {
    /// A reference to a Junction
    pub struct JunctionKey;
}

#[derive(Debug, Default, Clone)]
struct EmbeddingInner {
    junctions: Rc<RefCell<slotmap::SlotMap<JunctionKey, Junction>>>,
    next_voltage: u32,
    lines: slotmap::SlotMap<LineKey, Line>,
}

slotmap::new_key_type! {
    struct LineKey;
}

#[derive(Debug, Clone)]
struct Line {
    start: JunctionKey,
    middle: Option<Vector2i>,
    end: JunctionKey,
    junctions: Rc<RefCell<slotmap::SlotMap<JunctionKey, Junction>>>,
}

impl GodotConvert for JunctionKey {
    type Via = u64;
}
impl ToGodot for JunctionKey {
    fn to_godot(&self) -> Self::Via {
        self.data().as_ffi()
    }
}
impl FromGodot for JunctionKey {
    fn try_from_godot(key_data: Self::Via) -> Result<Self, ConvertError> {
        Ok(JunctionKey(slotmap::KeyData::from_ffi(key_data)))
    }
}

impl Embedding {
    // A list of all junctions in the circuit
    pub fn junctions(&self) -> Rc<RefCell<slotmap::SlotMap<JunctionKey, Junction>>> {
        self.inner.junctions()
    }

    // A list of all lines in the circuit
    pub fn lines(&self) -> Vec<Vec<Vector2i>> {
        self.inner.lines()
    }

    /// Add a new unoriented junction at the specified position
    pub fn add_junction(&mut self, junction: Vector2i) -> JunctionKey {
        self.inner.add_junction(junction)
    }

    /// Fetch a junction matching the key
    pub fn get_junction(&self, key: JunctionKey) -> Option<Junction> {
        self.inner.get_junction(key)
    }

    /// Add a connection between two junctions referenced by their indices
    pub fn add_connection(&mut self, start: JunctionKey, end: JunctionKey) {
        let mut start = start;
        let mut end = end;
        if start > end {
            // I enforce that end > start
            std::mem::swap(&mut start, &mut end);
        }
        if self
            .connections
            .iter()
            .any(|&connection| connection == (start, end))
            || start == end
        {
            return;
        }
        self.connections.push((start, end));
        self.inner.add_connection(start, end);
    }
}

impl EmbeddingInner {
    /// Add a new unoriented junction at the specified position
    fn add_junction(&mut self, position: Vector2i) -> JunctionKey {
        let candidate_junction = Junction {
            position,
            voltage: self.next_voltage,
            kind: (),
        };
        if let Some((key, _)) = self
            .junctions
            .borrow()
            .iter()
            .find(|(_, j)| j.position() == candidate_junction.position())
        {
            return key;
        }
        self.next_voltage += 1;
        self.junctions.borrow_mut().insert(candidate_junction)
    }

    fn get_junction(&self, key: JunctionKey) -> Option<Junction> {
        self.junctions.borrow().get(key).cloned()
    }

    fn add_connection(&mut self, start: JunctionKey, end: JunctionKey) {
        self.line_from(start, end);
        self.handle_intersections();
    }

    fn line_from(&mut self, start: JunctionKey, end: JunctionKey) {
        if start == end {
            return;
        }
        self.set_equal_voltage(start, end);
        let mut line = Line {
            start: std::cmp::min(start, end),
            middle: None,
            end: std::cmp::max(start, end),
            junctions: self.junctions.clone(),
        };

        if line.start().x != line.end().x && line.start().y != line.end().y {
            // this condition makes lines between points invariant wrt
            // which point is the start or end and also somewhat
            // prevents the program from generating swastikas
            let point = if line.start().x > line.end().x {
                Vector2i::new(line.start().x, line.end().y)
            } else {
                Vector2i::new(line.end().x, line.start().y)
            };
            line.middle = Some(point);
        }

        self.lines.insert(line);
    }

    fn set_equal_voltage(&mut self, start: JunctionKey, end: JunctionKey) {
        // connected junctions will be the same voltage, the minimum of
        // the subgraph
        let mut new_voltage = u32::MAX;
        let mut handled_junctions: HashSet<JunctionKey> = HashSet::default();
        let mut working_junctions = vec![start, end];
        while !working_junctions.is_empty() {
            // see if the voltage should be lower
            new_voltage = std::cmp::min(
                new_voltage,
                working_junctions
                    .iter()
                    .map(|&j| self.junctions.borrow()[j].voltage())
                    .max()
                    .expect("empty working junctions"),
            );
            // add the working junctions to handled
            for &junction in working_junctions.iter() {
                handled_junctions.insert(junction);
            }
            // get the next set of working junctions
            working_junctions = working_junctions
                .iter()
                .flat_map(|&junction| {
                    self.lines
                        .iter()
                        .filter_map(|(_, line)| {
                            if line.start == junction {
                                Some(line.end)
                            } else if line.end == junction {
                                Some(line.start)
                            } else {
                                None
                            }
                        })
                        .filter(|junction| !handled_junctions.contains(junction))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
        }
        for junction in handled_junctions {
            self.junctions.borrow_mut()[junction].set_voltage(new_voltage);
        }
    }

    fn handle_intersections(&mut self) {
        let new_junctions = self
            .lines
            .iter()
            .enumerate()
            .flat_map(|(i, (key, line))| {
                self.lines
                    .iter()
                    .skip(i + 1)
                    .flat_map(|(other_key, other_line)| {
                        if line.voltage() != other_line.voltage() {
                            return vec![];
                        }
                        line.intersect(other_line)
                            .iter()
                            .filter_map(|intersect| match intersect {
                                Intersection1D::<Vector2i>::Point(position) => {
                                    Some((key, other_key, *position))
                                }
                                Intersection1D::<Vector2i>::Segment(_start, _end) => None, // todo
                                Intersection1D::<Vector2i>::None => None,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut slice_points = slotmap::SecondaryMap::<LineKey, Vec<JunctionKey>>::new();
        for (line_key, other_line_key, junction) in new_junctions.iter() {
            let new_junction_key = self.add_junction(*junction);
            match slice_points.get_mut(*line_key) {
                Some(junctions) => junctions.push(new_junction_key),
                None => drop(slice_points.insert(*line_key, vec![new_junction_key])),
            }
            match slice_points.get_mut(*other_line_key) {
                Some(junctions) => junctions.push(new_junction_key),
                None => drop(slice_points.insert(*other_line_key, vec![new_junction_key])),
            }
        }
        for (line, junctions) in slice_points.iter() {
            let line = self.lines.get(line).expect("missing line");
            let (start, end) = (line.start, line.end);
            let mut previous = start;
            for junction in junctions {
                // I fear that there may be an issue with the ordering
                // junctions, but I haven't encountered it yet.
                self.line_from(previous, *junction);
                previous = *junction;
            }
            self.line_from(previous, end);
        }
        for line_key in slice_points.keys() {
            self.lines.remove(line_key);
        }
    }

    fn junctions(&self) -> Rc<RefCell<slotmap::SlotMap<JunctionKey, Junction>>> {
        self.junctions.clone()
    }

    fn lines(&self) -> Vec<Vec<Vector2i>> {
        self.lines
            .iter()
            .map(|(_, line)| line.coordinates())
            .collect()
    }
}

impl PartialEq for EmbeddingInner {
    fn eq(&self, other: &EmbeddingInner) -> bool {
        self.junctions.borrow().keys().collect::<HashSet<_>>()
            == other.junctions.borrow().keys().collect::<HashSet<_>>()
            && self
                .junctions
                .borrow()
                .keys()
                .all(|key| other.junctions.borrow().get(key) == self.junctions.borrow().get(key))
            && self.lines.keys().collect::<HashSet<_>>()
                == other.lines.keys().collect::<HashSet<_>>()
            && self
                .lines
                .keys()
                .all(|key| other.lines.get(key) == self.lines.get(key))
    }
}

impl Junction {
    pub fn position(&self) -> Vector2i {
        self.position
    }

    fn voltage(&self) -> u32 {
        self.voltage
    }

    fn set_voltage(&mut self, new_voltage: u32) {
        self.voltage = new_voltage;
    }
}

impl Line {
    fn intersect(&self, other: &Line) -> Vec<Intersection1D<Vector2i>> {
        let self_segments: Vec<(Vector2i, Vector2i)>;
        if let Some(middle) = self.middle {
            self_segments = vec![(self.start(), middle), (middle, self.end())];
        } else {
            self_segments = vec![(self.start(), self.end())];
        }
        let other_segments: Vec<(Vector2i, Vector2i)>;
        if let Some(middle) = other.middle {
            other_segments = vec![(other.start(), middle), (middle, other.end())];
        } else {
            other_segments = vec![(other.start(), other.end())];
        }

        self_segments
            .iter()
            .flat_map(|&self_segment| {
                other_segments
                    .iter()
                    .map(|&other_segment| segment_segment_intersection(self_segment, other_segment))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn start(&self) -> Vector2i {
        self.junctions
            .borrow()
            .get(self.start)
            .expect("hanging end reference")
            .position()
    }

    fn end(&self) -> Vector2i {
        self.junctions
            .borrow()
            .get(self.end)
            .expect("hanging start reference")
            .position()
    }

    fn coordinates(&self) -> Vec<Vector2i> {
        let mut vertices = vec![self.start()];
        if let Some(middle) = self.middle {
            vertices.push(middle);
        }
        vertices.push(self.end());
        vertices
    }

    fn voltage(&self) -> u32 {
        let voltage = self.junctions.borrow()[self.start].voltage();
        assert_eq!(voltage, self.junctions.borrow()[self.end].voltage());
        voltage
    }
}

impl PartialEq for Line {
    fn eq(&self, other: &Line) -> bool {
        self.junctions.borrow().keys().collect::<HashSet<_>>()
            == other.junctions.borrow().keys().collect::<HashSet<_>>()
            && self
                .junctions
                .borrow()
                .keys()
                .all(|key| other.junctions.borrow().get(key) == self.junctions.borrow().get(key))
            && self.start == other.start
            && self.middle == other.middle
            && self.end == other.end
    }
}

#[derive(Debug)]
enum Intersection1D<T> {
    Point(T),
    Segment(T, T),
    None,
}

/// Computes intersection of two line segments with the understanding
/// that the segments will be axis-aligned.
fn segment_segment_intersection(
    segment0: (Vector2i, Vector2i),
    segment1: (Vector2i, Vector2i),
) -> Intersection1D<Vector2i> {
    // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    let (Vector2i { x: x1, y: y1 }, Vector2i { x: x2, y: y2 }) = segment0;
    let (Vector2i { x: x3, y: y3 }, Vector2i { x: x4, y: y4 }) = segment1;
    if (x1 == x2 && x3 == x4) || (y1 == y2 && y3 == y4) {
        // parallel
        if x1 == x3 {
            // collinear in x
            return match interval_intersect((y1, y2), (y3, y4)) {
                Intersection1D::<i32>::Segment(start, end) => Intersection1D::<Vector2i>::Segment(
                    Vector2i::new(x1, start),
                    Vector2i::new(x1, end),
                ),
                Intersection1D::<i32>::Point(_) | Intersection1D::<i32>::None => {
                    Intersection1D::<Vector2i>::None
                }
            };
        }
        if y1 == y3 {
            // collinear in y
            return match interval_intersect((y1, y2), (y3, y4)) {
                Intersection1D::<i32>::Segment(start, end) => Intersection1D::<Vector2i>::Segment(
                    Vector2i::new(start, y1),
                    Vector2i::new(end, y1),
                ),
                Intersection1D::<i32>::Point(_) | Intersection1D::<i32>::None => {
                    Intersection1D::<Vector2i>::None
                }
            };
        }
        return Intersection1D::<Vector2i>::None;
    }

    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    // denominator should only be 0 if the lines are parallel, which has
    // already been handled
    assert_ne!(denominator, 0, "mysteriously parallel");
    let t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    let u_numerator = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3);
    let t = t_numerator as f32 / denominator as f32;
    let u = -(u_numerator as f32 / denominator as f32);
    if t_u_inbounds(t, u) {
        // point intersection
        return Intersection1D::<Vector2i>::Point(Vector2i::new(
            (x1 as f32 + t * (x2 - x1) as f32).round() as i32,
            (y1 as f32 + t * (y2 - y1) as f32).round() as i32,
        ));
    }
    Intersection1D::<Vector2i>::None
}

fn interval_intersect(mut a: (i32, i32), mut b: (i32, i32)) -> Intersection1D<i32> {
    // https://scicomp.stackexchange.com/a/26260
    if a.0 > a.1 {
        a = (a.1, a.0);
    }
    if b.0 > b.1 {
        b = (b.1, b.0);
    }
    if b.0 > a.1 || a.0 > b.1 {
        return Intersection1D::<i32>::None;
    }
    if b.0 == a.1 {
        return Intersection1D::<i32>::Point(a.1);
    }
    if a.0 == b.1 {
        return Intersection1D::<i32>::Point(a.0);
    }
    let start = std::cmp::max(a.0, b.0);
    let end = std::cmp::min(a.1, b.1);
    Intersection1D::<i32>::Segment(start, end)
}

/// Checks if t and u are in the inverval [0, 1], but not both at endpoints
///
/// If they're both at endpoints, then you don't have an intersection.  You
/// just have a multisegment line.
fn t_u_inbounds(t: f32, u: f32) -> bool {
    if !((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)) {
        return false;
    }
    if (t == 0.0 || t == 1.0) && (u == 0.0 || u == 1.0) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_junctions_adds_junctions() {
        let mut embedding: Embedding = Default::default();
        embedding.add_junction(Vector2i::new(0, 0));
        embedding.add_junction(Vector2i::new(10, 0));
        embedding.add_junction(Vector2i::new(0, 10));
        embedding.add_junction(Vector2i::new(-10, 0));
        embedding.add_junction(Vector2i::new(0, -10));
        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(0, 0),
                Vector2i::new(10, 0),
                Vector2i::new(0, 10),
                Vector2i::new(-10, 0),
                Vector2i::new(0, -10),
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        )
    }

    #[test]
    fn lines_between_single_axis_unoriented_junctions() {
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(0, 0));
        let j1 = embedding.add_junction(Vector2i::new(10, 0));
        let j2 = embedding.add_junction(Vector2i::new(0, 10));
        let j3 = embedding.add_junction(Vector2i::new(-10, 0));
        let j4 = embedding.add_junction(Vector2i::new(0, -10));
        embedding.add_connection(j0, j1);
        embedding.add_connection(j0, j2);
        embedding.add_connection(j0, j3);
        embedding.add_connection(j0, j4);
        embedding.add_connection(j0, j0); // self connections should be rejected
        embedding.add_connection(j0, j1); // duplicates should be rejected
        embedding.add_connection(j2, j0); // reverse order dupe should be rejected

        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(0, 0),
                Vector2i::new(10, 0),
                Vector2i::new(0, 10),
                Vector2i::new(-10, 0),
                Vector2i::new(0, -10),
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().into_iter().collect::<HashSet<_>>(),
            vec![
                vec![Vector2i::new(0, 0), Vector2i::new(10, 0)],
                vec![Vector2i::new(0, 0), Vector2i::new(0, 10)],
                vec![Vector2i::new(0, 0), Vector2i::new(-10, 0)],
                vec![Vector2i::new(0, 0), Vector2i::new(0, -10)]
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn lines_between_two_axis_unoriented_junctions() {
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(0, 0));
        let j1 = embedding.add_junction(Vector2i::new(10, 10));
        let j2 = embedding.add_junction(Vector2i::new(-10, 10));
        let j3 = embedding.add_junction(Vector2i::new(-10, -10));
        let j4 = embedding.add_junction(Vector2i::new(10, -10));

        let mut forward_embedding = embedding.clone();
        forward_embedding.add_connection(j0, j1);
        forward_embedding.add_connection(j0, j2);
        forward_embedding.add_connection(j0, j3);
        forward_embedding.add_connection(j0, j4);
        let mut backward_embedding = embedding;
        backward_embedding.add_connection(j1, j0);
        backward_embedding.add_connection(j2, j0);
        backward_embedding.add_connection(j3, j0);
        backward_embedding.add_connection(j4, j0);
        // the embedding should not change based on the direction the
        // connections are made
        assert_eq!(forward_embedding, backward_embedding);
        assert_eq!(
            forward_embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(0, 0),
                Vector2i::new(10, 10),
                Vector2i::new(-10, 10),
                Vector2i::new(-10, -10),
                Vector2i::new(10, -10),
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            forward_embedding.lines(),
            vec![
                vec![
                    Vector2i::new(0, 0),
                    Vector2i::new(10, 0),
                    Vector2i::new(10, 10)
                ],
                vec![
                    Vector2i::new(0, 0),
                    Vector2i::new(0, 10),
                    Vector2i::new(-10, 10)
                ],
                vec![
                    Vector2i::new(0, 0),
                    Vector2i::new(0, -10),
                    Vector2i::new(-10, -10)
                ],
                vec![
                    Vector2i::new(0, 0),
                    Vector2i::new(10, 0),
                    Vector2i::new(10, -10)
                ]
            ]
        );
    }

    #[test]
    fn perpendicular_intersection() {
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(10, 0)); // no origin!
        let j1 = embedding.add_junction(Vector2i::new(0, 10));
        let j2 = embedding.add_junction(Vector2i::new(-10, 0));
        let j3 = embedding.add_junction(Vector2i::new(0, -10));
        embedding.add_connection(j0, j1); // ensure the next connections are the same voltage
        embedding.add_connection(j0, j2);
        embedding.add_connection(j1, j3);
        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(10, 0),
                Vector2i::new(0, 10),
                Vector2i::new(-10, 0),
                Vector2i::new(0, -10),
                Vector2i::new(0, 0), // origin!
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().iter().collect::<HashSet<_>>(),
            vec![
                vec![
                    Vector2i::new(10, 0),
                    Vector2i::new(10, 10),
                    Vector2i::new(0, 10)
                ],
                vec![Vector2i::new(10, 0), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, 10), Vector2i::new(0, 0)],
                vec![Vector2i::new(-10, 0), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, -10), Vector2i::new(0, 0)]
            ]
            .iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn perpendicular_intersect_wild_numbers() {
        // This is a case that gave me trouble during initial development
        // The intersection calculation was hitting an assert to prevent
        // divide-by-zero in a case where I worked out it shouldn't.
        // Turns out I wasn't handling parallel non-colinear cases.
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(323, 321));
        let j1 = embedding.add_junction(Vector2i::new(782, 342));
        let j2 = embedding.add_junction(Vector2i::new(534, 153));
        let j3 = embedding.add_junction(Vector2i::new(541, 514));
        embedding.add_connection(j1, j2); // voltage connection
        embedding.add_connection(j0, j1);
        embedding.add_connection(j2, j3);
        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(323, 321),
                Vector2i::new(782, 342),
                Vector2i::new(534, 153),
                Vector2i::new(541, 514),
                Vector2i::new(782, 321), // intersection w/ voltage connection
                Vector2i::new(541, 153), // intersection w/ voltage connection
                Vector2i::new(541, 321),
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn intersection_through_two_lines() {
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(0, -10));
        let j1 = embedding.add_junction(Vector2i::new(0, 10));
        let j2 = embedding.add_junction(Vector2i::new(10, 10));
        let j3 = embedding.add_junction(Vector2i::new(10, -10));
        let j4 = embedding.add_junction(Vector2i::new(-10, 0));
        let j5 = embedding.add_junction(Vector2i::new(20, 0));
        embedding.add_connection(j1, j2); // voltage connection
        embedding.add_connection(j2, j5); // voltage connection
        embedding.add_connection(j0, j1);
        embedding.add_connection(j2, j3);
        embedding.add_connection(j4, j5); // This is like the horizontal on an H

        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(0, -10),
                Vector2i::new(0, 10),
                Vector2i::new(10, 10),
                Vector2i::new(10, -10),
                Vector2i::new(-10, 0),
                Vector2i::new(20, 0),
                Vector2i::new(0, 0),
                Vector2i::new(10, 0)
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().iter().collect::<HashSet<_>>(),
            vec![
                vec![Vector2i::new(0, 10), Vector2i::new(10, 10)],
                vec![
                    Vector2i::new(10, 10),
                    Vector2i::new(20, 10),
                    Vector2i::new(20, 0)
                ],
                vec![Vector2i::new(0, -10), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, 10), Vector2i::new(0, 0)],
                vec![Vector2i::new(-10, 0), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, 0), Vector2i::new(10, 0)],
                vec![Vector2i::new(20, 0), Vector2i::new(10, 0)],
                vec![Vector2i::new(10, 10), Vector2i::new(10, 0)],
                vec![Vector2i::new(10, -10), Vector2i::new(10, 0)],
            ]
            .iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn different_voltage_lines_dont_intersect() {
        // this is the same as perpendicular_intersection(), but without
        // the voltage connection
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(10, 0)); // no origin!
        let j1 = embedding.add_junction(Vector2i::new(0, 10));
        let j2 = embedding.add_junction(Vector2i::new(-10, 0));
        let j3 = embedding.add_junction(Vector2i::new(0, -10));
        // the connections don't have any junctions in common, so different voltage
        embedding.add_connection(j0, j2);
        embedding.add_connection(j1, j3);
        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(10, 0),
                Vector2i::new(0, 10),
                Vector2i::new(-10, 0),
                Vector2i::new(0, -10), // still no origin!
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().iter().collect::<HashSet<_>>(),
            vec![
                vec![Vector2i::new(10, 0), Vector2i::new(-10, 0)],
                vec![Vector2i::new(0, 10), Vector2i::new(0, -10)],
            ]
            .iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn connections_after_bridging_intersect() {
        let mut embedding: Embedding = Default::default();
        let test0 = embedding.add_junction(Vector2i::new(-20, 0));
        let j0 = embedding.add_junction(Vector2i::new(-20, 10));
        let bridge0 = embedding.add_junction(Vector2i::new(-10, 10));
        let j1 = embedding.add_junction(Vector2i::new(-10, -10));
        let test1 = embedding.add_junction(Vector2i::new(20, 0));
        let j2 = embedding.add_junction(Vector2i::new(20, 10));
        let bridge1 = embedding.add_junction(Vector2i::new(10, 10));
        let j3 = embedding.add_junction(Vector2i::new(10, -10));
        embedding.add_connection(test0, j0); // left side U
        embedding.add_connection(j0, bridge0);
        embedding.add_connection(bridge0, j1);
        embedding.add_connection(test1, j2); // right side U
        embedding.add_connection(j2, bridge1);
        embedding.add_connection(bridge1, j3);
        embedding.add_connection(bridge0, bridge1); // bridge
        embedding.add_connection(test0, test1); // should cause 2 intersections

        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(-20, 0),
                Vector2i::new(-20, 10),
                Vector2i::new(-10, 10),
                Vector2i::new(-10, -10),
                Vector2i::new(20, 0),
                Vector2i::new(20, 10),
                Vector2i::new(10, 10),
                Vector2i::new(10, -10),
                Vector2i::new(-10, 0),
                Vector2i::new(10, 0),
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().iter().collect::<HashSet<_>>(),
            vec![
                vec![Vector2i::new(-20, 0), Vector2i::new(-20, 10)],
                vec![Vector2i::new(-20, 10), Vector2i::new(-10, 10)],
                vec![Vector2i::new(-10, 10), Vector2i::new(-10, 0)],
                vec![Vector2i::new(-10, -10), Vector2i::new(-10, 0)],
                vec![Vector2i::new(20, 0), Vector2i::new(20, 10)],
                vec![Vector2i::new(20, 10), Vector2i::new(10, 10)],
                vec![Vector2i::new(10, 10), Vector2i::new(10, 0)],
                vec![Vector2i::new(10, -10), Vector2i::new(10, 0)],
                vec![Vector2i::new(-10, 10), Vector2i::new(10, 10)],
                vec![Vector2i::new(-20, 0), Vector2i::new(-10, 0)],
                vec![Vector2i::new(-10, 0), Vector2i::new(10, 0)],
                vec![Vector2i::new(20, 0), Vector2i::new(10, 0)],
            ]
            .iter()
            .collect::<HashSet<_>>()
        );
    }

    #[test]
    fn intersection_created_when_voltage_connects() {
        // this is the same as perpendicular_intersection(), but the
        // voltage connection is added last
        let mut embedding: Embedding = Default::default();
        let j0 = embedding.add_junction(Vector2i::new(10, 0)); // no origin!
        let j1 = embedding.add_junction(Vector2i::new(0, 10));
        let j2 = embedding.add_junction(Vector2i::new(-10, 0));
        let j3 = embedding.add_junction(Vector2i::new(0, -10));
        embedding.add_connection(j0, j2);
        embedding.add_connection(j1, j3);
        embedding.add_connection(j0, j1);

        assert_eq!(
            embedding
                .junctions()
                .borrow()
                .iter()
                .map(|(_, junction)| junction.position())
                .collect::<HashSet<_>>(),
            vec![
                Vector2i::new(10, 0),
                Vector2i::new(0, 10),
                Vector2i::new(-10, 0),
                Vector2i::new(0, -10),
                Vector2i::new(0, 0), // origin!
            ]
            .into_iter()
            .collect::<HashSet<_>>()
        );
        assert_eq!(
            embedding.lines().iter().collect::<HashSet<_>>(),
            vec![
                vec![
                    Vector2i::new(10, 0),
                    Vector2i::new(10, 10),
                    Vector2i::new(0, 10)
                ],
                vec![Vector2i::new(10, 0), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, 10), Vector2i::new(0, 0)],
                vec![Vector2i::new(-10, 0), Vector2i::new(0, 0)],
                vec![Vector2i::new(0, -10), Vector2i::new(0, 0)]
            ]
            .iter()
            .collect::<HashSet<_>>()
        );
    }
}
