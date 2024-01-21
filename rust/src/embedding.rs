use godot::builtin::Vector2i;

#[derive(Clone, Debug, Default)]
pub struct Embedding {
    junctions: Vec<Vector2i>,
    connections: Vec<(usize, usize)>,
    lines: Vec<Vec<Vector2i>>,
}

impl Embedding {
    pub fn junctions(&self) -> &Vec<Vector2i> {
        &self.junctions
    }

    pub fn lines(&self) -> &Vec<Vec<Vector2i>> {
        &self.lines
    }

    pub fn add_junction(&self, junction: Vector2i) -> Self {
        let mut new = self.clone();
        new.junctions.push(junction);
        new
    }

    pub fn add_connection(&self, start: usize, end: usize) -> Self {
        let mut new = self.clone();
        if new
            .connections
            .iter()
            .any(|&connection| connection == (start, end))
        {
            return new;
        }
        new.connections.push((start, end));
        // unwrap() is ok because the last element is added the line before
        new.lines
            .push(new.line_from(*new.connections.last().unwrap()));
        new
    }

    fn line_from(&self, connection: (usize, usize)) -> Vec<Vector2i> {
        let start = self.junctions[connection.0];
        let end = self.junctions[connection.1];
        let mut points = vec![start];
        if start.x == end.x || start.y == end.y {
            points.push(start);
            points.push(end);
        } else {
            points.push(start);
            if start.x > end.x {
                points.push(Vector2i::new(start.x, end.y));
            } else {
                points.push(Vector2i::new(end.x, start.y));
            }
            points.push(end);
        }
        points
    }
}
