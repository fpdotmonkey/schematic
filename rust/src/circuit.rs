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
    proto_line: Option<embedding::JunctionKey>,
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
    key: embedding::JunctionKey,
    click_size: u32,
    highlight: Color,
    base: Base<Sprite2D>,
}

#[derive(GodotClass)]
#[class(init, base=Line2D)]
struct CircuitLine {
    points: Vec<Vector2i>,
    base: Base<Line2D>,
}

#[derive(GodotClass)]
#[class(init, base=Line2D)]
struct CircuitProtoLine {
    start: embedding::JunctionKey,
    end: Vector2i,
    base: Base<Line2D>,
}

#[godot_api]
impl INode2D for Circuit {
    fn unhandled_input(&mut self, event: Gd<InputEvent>) {
        let Ok(event) = event.try_cast::<InputEventMouseButton>() else {
            return;
        };
        let position = Vector2i::from_vector2(event.get_position());

        if !event.is_pressed() {
            if self.proto_line.is_some() {
                // line ending
                godot_print!("cancel line");
                self.proto_line = None;
                self.base_mut().queue_redraw();
            } else {
                // junction creation
                godot_print!("create junction at {}", position);
                self.embedding.add_junction(position);
                self.base_mut().queue_redraw();
                let mut viewport = self.base().get_viewport().expect("missing viewport");
                viewport.set_input_as_handled()
            }
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
            .borrow()
            .iter()
            .map(|(key, junction)| {
                let mut junction_node: Gd<CircuitJunction> =
                    Gd::from_init_fn(|base| CircuitJunction {
                        key,
                        click_size: self.junction_click_size,
                        highlight: self.junction_highlight,
                        base,
                    });
                junction_node.connect(
                    "junction_was_selected".into(),
                    Callable::from_object_method(
                        &self.base().to_godot(),
                        StringName::from("on_junction_selected"),
                    ),
                );
                junction_node.connect(
                    "junction_was_unselected".into(),
                    Callable::from_object_method(
                        &self.base().to_godot(),
                        StringName::from("on_junction_unselected"),
                    ),
                );
                junction_node.set_position(Vector2::from_vector2i(junction.position()));
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
        if let Some(proto_line) = self.proto_line {
            let proto_line_node = Gd::from_init_fn(|base| CircuitProtoLine {
                start: proto_line,
                end: Default::default(),
                base,
            });
            // CircuitProtoLine::new_alloc();
            // proto_line_node
            // .bind_mut()
            // .set_start(self.embedding.junctions()[proto_line_idx]);
            self.base_mut().add_child(proto_line_node.clone().upcast());
        }
    }
}

#[godot_api]
impl Circuit {
    #[func]
    fn on_junction_selected(&mut self, key: Variant) {
        if self.proto_line.is_none() {
            godot_print!("start a line");
            let key = embedding::JunctionKey::from_variant(&key);
            self.proto_line = Some(key);
            self.base_mut().queue_redraw();
        }
    }
    #[func]
    fn on_junction_unselected(&mut self, key: Variant) {
        if let Some(proto_line) = self.proto_line {
            godot_print!("end line");
            let key = embedding::JunctionKey::from_variant(&key);
            self.embedding.add_connection(proto_line, key);
            self.proto_line = None;
            self.base_mut().queue_redraw();
        }
    }
}

#[godot_api]
impl ISprite2D for CircuitJunction {
    fn ready(&mut self) {
        let texture: Gd<Texture2D> = load("res://icon.svg");
        self.base_mut().set_texture(texture);
    }

    fn unhandled_input(&mut self, event: Gd<InputEvent>) {
        if let Ok(event) = event.clone().try_cast::<InputEventMouseButton>() {
            if self.is_over_junction(event.get_position()) {
                let key = self.key.to_variant();
                if event.is_pressed() && !event.is_echo() {
                    self.base_mut()
                        .emit_signal("junction_was_selected".into(), &[key]);
                } else if !event.is_pressed() {
                    self.base_mut()
                        .emit_signal("junction_was_unselected".into(), &[key]);
                }
                let mut viewport = self.base().get_viewport().expect("missing viewport");
                viewport.set_input_as_handled()
            }
        }
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

#[godot_api]
impl CircuitJunction {
    #[signal]
    fn junction_was_selected(key: Variant);
    #[signal]
    fn junction_was_unselected(key: Variant);
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
    fn input(&mut self, event: Gd<InputEvent>) {
        let Ok(event) = event.try_cast::<InputEventMouseMotion>() else {
            return;
        };
        self.end = Vector2i::from_vector2(event.get_position());
        let mut points = PackedVector2Array::new();
        let start_position = self
            .base()
            .get_parent()
            .unwrap()
            .cast::<Circuit>()
            .bind()
            .embedding
            .get_junction(self.start)
            .expect("hanging start reference")
            .position();
        points.push(Vector2::from_vector2i(start_position));
        points.push(Vector2::from_vector2i(self.end));
        self.base_mut().set_points(points);
        self.base_mut().queue_redraw();
    }
}
