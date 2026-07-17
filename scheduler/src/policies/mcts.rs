use rand::{Rng, rngs::StdRng};

use crate::{
    BranchId, BranchState, BranchTree, Command, EngineEvent, MAX_BRANCH_WIDTH, MctsConfig, Policy,
    SchedulerError,
};

#[derive(Clone, Debug)]
pub struct MctsPolicy {
    expansion_width: u32,
    max_depth: u32,
    exploration_weight: f64,
}

impl MctsPolicy {
    pub fn new(config: MctsConfig) -> Result<Self, SchedulerError> {
        if config.expansion_width == 0 || config.expansion_width > MAX_BRANCH_WIDTH {
            return Err(SchedulerError::InvalidConfig(
                "MCTS expansion_width must be between 1 and MAX_BRANCH_WIDTH",
            ));
        }
        if config.max_depth == 0 {
            return Err(SchedulerError::InvalidConfig(
                "MCTS max_depth must be greater than zero",
            ));
        }
        if !config.exploration_weight.is_finite() || config.exploration_weight < 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "MCTS exploration_weight must be finite and non-negative",
            ));
        }
        Ok(Self {
            expansion_width: config.expansion_width,
            max_depth: config.max_depth,
            exploration_weight: config.exploration_weight,
        })
    }

    fn choose_child(
        &self,
        tree: &BranchTree,
        parent: BranchId,
        rng: &mut StdRng,
    ) -> Option<BranchId> {
        let mut children: Vec<_> = tree
            .children(parent)
            .ok()?
            .iter()
            .copied()
            .filter(|child| tree.get(*child).is_some_and(|node| node.state().is_live()))
            .collect();
        children.sort_unstable();
        if children.is_empty() {
            return None;
        }

        let unvisited: Vec<_> = children
            .iter()
            .copied()
            .filter(|child| tree.get(*child).expect("known child").visits() == 0)
            .collect();
        if !unvisited.is_empty() {
            return Some(unvisited[rng.random_range(0..unvisited.len())]);
        }

        let parent_visits = tree.get(parent).expect("known parent").visits().max(1) as f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut best = Vec::new();
        for child in children {
            let node = tree.get(child).expect("known child");
            let visits = node.visits() as f64;
            let exploitation = node.value_sum() / visits;
            let exploration = self.exploration_weight * (parent_visits.ln() / visits).sqrt();
            let score = exploitation + exploration;
            match score.total_cmp(&best_score) {
                std::cmp::Ordering::Greater => {
                    best_score = score;
                    best.clear();
                    best.push(child);
                }
                std::cmp::Ordering::Equal => best.push(child),
                std::cmp::Ordering::Less => {}
            }
        }
        Some(best[rng.random_range(0..best.len())])
    }

    fn select(
        &self,
        tree: &mut BranchTree,
        rng: &mut StdRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        let mut current = tree.root();
        loop {
            let node = tree
                .get(current)
                .ok_or(SchedulerError::UnknownBranch(current))?;
            match node.state() {
                BranchState::Active if node.depth() < self.max_depth => {
                    let children = tree.fork(current, self.expansion_width)?;
                    let selected = children[rng.random_range(0..children.len())];
                    return Ok(vec![
                        Command::ForkAt {
                            branch: current,
                            width: self.expansion_width,
                        },
                        Command::Continue { branch: selected },
                    ]);
                }
                BranchState::Active => {
                    return Ok(vec![Command::Continue { branch: current }]);
                }
                BranchState::Expanded => {
                    let Some(child) = self.choose_child(tree, current, rng) else {
                        return Ok(Vec::new());
                    };
                    current = child;
                }
                BranchState::Killed | BranchState::Finalized => return Ok(Vec::new()),
            }
        }
    }
}

impl Policy for MctsPolicy {
    fn on_event(
        &mut self,
        event: &EngineEvent,
        tree: &mut BranchTree,
        rng: &mut StdRng,
    ) -> Result<Vec<Command>, SchedulerError> {
        if matches!(
            event,
            EngineEvent::TokenSampled { .. } | EngineEvent::ValueScored { .. }
        ) {
            let branch = event.branch();
            let node = tree
                .get(branch)
                .ok_or(SchedulerError::UnknownBranch(branch))?;
            if !node.has_value() {
                return Ok(Vec::new());
            }
            let value = node.value_estimate();
            tree.backpropagate(branch, value)?;
        }
        self.select(tree, rng)
    }
}
