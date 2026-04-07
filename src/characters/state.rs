use bevy::prelude::*;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CharacterState {
    #[default]
    Idle,
    Walking,
    Running,
    Jumping,
}

impl CharacterState {
    pub fn is_grounded(&self) -> bool {
        matches!(
            self,
            CharacterState::Idle | CharacterState::Walking | CharacterState::Running
        )
    }
}
