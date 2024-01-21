use godot::prelude::*;

mod circuit;
mod embedding;

struct SchematicLibrary;

#[gdextension]
unsafe impl ExtensionLibrary for SchematicLibrary {}
