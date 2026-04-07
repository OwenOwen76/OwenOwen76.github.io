use bevy::prelude::*;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Facing {
    Up,
    Left,
    #[default]
    Down,
    Right,
}

impl Facing {
    pub fn from_velocity(velocity: Vec2) -> Self {
        if velocity.x.abs() > velocity.y.abs() {
            if velocity.x > 0.0 {
                Facing::Right
            } else {
                Facing::Left
            }
        } else {
            if velocity.y > 0.0 {
                Facing::Up
            } else {
                Facing::Down
            }
        }
    }

    pub(crate) fn direction_index(self) -> usize {
        match self {
            Facing::Up => 0,
            Facing::Left => 1,
            Facing::Down => 2,
            Facing::Right => 3,
        }
    }
}
