use crate::embedding;
use godot::engine::{
    ILine2D, ISprite2D, InputEvent, InputEventMouseButton, InputEventMouseMotion, Line2D, Sprite2D,
    Texture2D,
};
use godot::prelude::*;

#[derive(GodotClass)]
#[class(init, base=Node2D)]
struct Circuit {
    #[export]
    #[init(default = 2)]
    junction_click_size: u32,
    #[export]
    #[init(default = Color::from_rgb(1.0, 1.0, 0.0))]
    junction_highlight: Color,
    embedding: embedding::Embedding,
    /// The start of a line in progress
    proto_line: Option<usize>,
    #[base]
    base: Base<Node2D>,
}

#[derive(PartialEq)]
struct Line {
    start: usize,
    end: usize,
}

#[derive(GodotClass)]
#[class(init, base=Sprite2D)]
struct CircuitJunction {
    click_size: u32,
    highlight: Color,
    #[base]
    base: Base<Sprite2D>,
}

#[derive(GodotClass)]
#[class(init, base=Line2D)]
struct CircuitLine {
    points: Vec<Vector2i>,
    #[base]
    base: Base<Line2D>,
}

#[derive(GodotClass)]
#[class(base=Line2D)]
struct CircuitProtoLine {
    #[export]
    start: Vector2i,
    end: Vector2i,
    #[base]
    base: Base<Line2D>,
}

#[godot_api]
impl INode2D for Circuit {
    fn unhandled_input(&mut self, event: Gd<InputEvent>) {
        let Ok(event) = event.try_cast::<InputEventMouseButton>() else {
            return;
        };
        let position = Vector2i::from_vector2(event.get_position());

        if let Some(proto_line) = self.proto_line {
            // line ending
            if !event.is_pressed() {
                if let Some((i, _)) =
                    self.embedding
                        .junctions()
                        .iter()
                        .enumerate()
                        .find(|(_, &junction)| {
                            (junction - position).length_squared()
                                <= self.junction_click_size.pow(2).into()
                        })
                {
                    godot_print!("end line at {}", position);
                    self.embedding = self.embedding.add_connection(proto_line, i);
                } else {
                    godot_print!("cancel line");
                }
                self.proto_line = None;
                self.base_mut().queue_redraw();
                return;
            }
        }

        if let Some((i, _)) =
            self.embedding
                .junctions()
                .iter()
                .enumerate()
                .find(|(_, &junction)| {
                    (junction - position).length_squared() <= self.junction_click_size.pow(2).into()
                })
        {
            // line starting
            if event.is_pressed() {
                godot_print!("start a line at {}", position);
                self.proto_line = Some(i);
                self.base_mut().queue_redraw();
            }
        } else if !event.is_pressed() {
            // junction creation
            godot_print!("create junction at {}", position);
            self.embedding = self.embedding.add_junction(position);
            self.base_mut().queue_redraw();
        }
    }

    fn draw(&mut self) {
        // out with the old
        for mut child in self.base().get_children().iter_shared() {
            child.queue_free();
        }
        // and in with the new
        let junction_nodes = self
            .embedding
            .junctions()
            .iter()
            .map(|position| {
                let mut junction_node: Gd<CircuitJunction> =
                    Gd::from_init_fn(|base| CircuitJunction {
                        click_size: self.junction_click_size,
                        highlight: self.junction_highlight,
                        base,
                    });
                junction_node.set_position(Vector2::from_vector2i(*position));
                junction_node
            })
            .collect::<Vec<_>>();
        for junction_node in junction_nodes.iter() {
            self.base_mut().add_child(junction_node.clone().upcast());
        }
        let line_nodes = self
            .embedding
            .lines()
            .iter()
            .map(|points| {
                Gd::from_init_fn(|base| CircuitLine {
                    points: points.to_vec(),
                    base,
                })
            })
            .collect::<Vec<_>>();
        for line_node in line_nodes.iter() {
            self.base_mut().add_child(line_node.clone().upcast());
        }
        if let Some(proto_line_idx) = self.proto_line {
            let mut proto_line_node = CircuitProtoLine::new_alloc();
            proto_line_node
                .bind_mut()
                .set_start(self.embedding.junctions()[proto_line_idx]);
            self.base_mut().add_child(proto_line_node.clone().upcast());
        }
    }
}

#[godot_api]
impl ISprite2D for CircuitJunction {
    fn ready(&mut self) {
        let texture: Gd<Texture2D> = load("res://icon.svg");
        self.base_mut().set_texture(texture);
    }

    fn input(&mut self, event: Gd<InputEvent>) {
        let Ok(event) = event.try_cast::<InputEventMouseMotion>() else {
            return;
        };
        if self.is_over_junction(event.get_position()) {
            let highlight = self.highlight;
            self.base_mut().set_modulate(highlight);
        } else {
            self.base_mut().set_modulate(Color::WHITE);
        }
    }
}

impl CircuitJunction {
    fn is_over_junction(&self, point: Vector2) -> bool {
        (point - self.base().get_position()).length_squared()
            <= (self.click_size * self.click_size) as f32
    }
}

#[godot_api]
impl ILine2D for CircuitLine {
    fn ready(&mut self) {
        let points = self
            .points
            .iter()
            .map(|&point| Vector2::from_vector2i(point))
            .collect();
        self.base_mut().set_points(points);
    }
}

#[godot_api]
impl ILine2D for CircuitProtoLine {
    fn init(base: Base<Line2D>) -> Self {
        Self {
            start: Vector2i { x: 0, y: 0 },
            end: Vector2i { x: 0, y: 0 },
            base,
        }
    }

    fn input(&mut self, event: Gd<InputEvent>) {
        let Ok(event) = event.try_cast::<InputEventMouseMotion>() else {
            return;
        };
        self.end = Vector2i::from_vector2(event.get_position());
        let mut points = PackedVector2Array::new();
        points.push(Vector2::from_vector2i(self.start));
        points.push(Vector2::from_vector2i(self.end));
        self.base_mut().set_points(points);
        self.base_mut().queue_redraw();
    }
}
