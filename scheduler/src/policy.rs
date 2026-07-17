use rand::rngs::StdRng;

use crate::{BranchTree, Command, EngineEvent, SchedulerError};

use crate::policies::{BeamPolicy, BestFirstPolicy, MctsPolicy};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeamConfig {
    pub width: u32,
    pub fork_width: u32,
    pub fork_at_tokens: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BestFirstConfig {
    pub expansion_width: u32,
    pub max_depth: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsConfig {
    pub expansion_width: u32,
    pub max_depth: u32,
    pub exploration_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyConfig {
    Beam(BeamConfig),
    BestFirst(BestFirstConfig),
    Mcts(MctsConfig),
}

impl PolicyConfig {
    pub(crate) fn build(&self) -> Result<Box<dyn Policy>, SchedulerError> {
        match self {
            Self::Beam(config) => Ok(Box::new(BeamPolicy::new(config.clone())?)),
            Self::BestFirst(config) => Ok(Box::new(BestFirstPolicy::new(config.clone())?)),
            Self::Mcts(config) => Ok(Box::new(MctsPolicy::new(config.clone())?)),
        }
    }
}

/// Pluggable deterministic branch policy.
pub trait Policy: Send + Sync {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        rng: &mut StdRng,
    ) -> Result<Vec<Command>, SchedulerError>;
}
