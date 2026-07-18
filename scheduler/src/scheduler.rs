use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::{SeedableRng, rngs::StdRng};

use crate::{
    BranchId, BranchState, BranchTree, BudgetController, Command, EngineEvent, KillReason,
    LogprobScorer, Policy, PolicyConfig, SchedulerError, ValueScorer,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub policy: PolicyConfig,
    pub seed: u64,
    pub total_token_budget: u64,
    pub per_branch_token_budget: u64,
    pub speculative_kill_margin: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingBudgetTerminal {
    Branch,
    Tree,
}

/// Pure event-driven scheduler. It performs no I/O and owns all deterministic state.
pub struct Scheduler {
    tree: BranchTree,
    budget: BudgetController,
    policy: Box<dyn Policy>,
    scorer: Option<Box<dyn ValueScorer>>,
    rng: StdRng,
    command_queue: VecDeque<Command>,
    outstanding_continuations: BTreeSet<BranchId>,
    pending_external_values: BTreeSet<BranchId>,
    pending_budget_terminals: BTreeMap<BranchId, PendingBudgetTerminal>,
    speculative_kill_margin: Option<f64>,
    rollback_custom_policy_errors: bool,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        Self::with_scorer(config, Box::<LogprobScorer>::default())
    }

    pub fn with_scorer(
        config: SchedulerConfig,
        scorer: Box<dyn ValueScorer>,
    ) -> Result<Self, SchedulerError> {
        let policy = config.policy.build()?;
        Self::from_components(config, policy, Some(scorer), false)
    }

    /// Builds a scheduler whose engine supplies one `ValueScored` event after each token.
    pub fn with_external_values(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        let policy = config.policy.build()?;
        Self::from_components(config, policy, None, false)
    }

    pub fn with_components(
        config: SchedulerConfig,
        policy: Box<dyn Policy>,
        scorer: Box<dyn ValueScorer>,
    ) -> Result<Self, SchedulerError> {
        Self::from_components(config, policy, Some(scorer), true)
    }

    fn from_components(
        config: SchedulerConfig,
        policy: Box<dyn Policy>,
        scorer: Option<Box<dyn ValueScorer>>,
        rollback_custom_policy_errors: bool,
    ) -> Result<Self, SchedulerError> {
        if config
            .speculative_kill_margin
            .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
        {
            return Err(SchedulerError::InvalidConfig(
                "speculative_kill_margin must be finite and non-negative",
            ));
        }
        let budget =
            BudgetController::new(config.total_token_budget, config.per_branch_token_budget)?;
        Ok(Self {
            tree: BranchTree::new(),
            budget,
            policy,
            scorer,
            rng: StdRng::seed_from_u64(config.seed),
            command_queue: VecDeque::new(),
            outstanding_continuations: BTreeSet::new(),
            pending_external_values: BTreeSet::new(),
            pending_budget_terminals: BTreeMap::new(),
            speculative_kill_margin: config.speculative_kill_margin,
            rollback_custom_policy_errors,
        })
    }

    #[must_use]
    pub const fn tree(&self) -> &BranchTree {
        &self.tree
    }

    #[must_use]
    pub const fn budget(&self) -> &BudgetController {
        &self.budget
    }

    pub fn feed_event(&mut self, event: EngineEvent) -> Result<(), SchedulerError> {
        if !self.rollback_custom_policy_errors {
            return self.feed_event_inner(event);
        }
        let tree = self.tree.clone();
        let budget = self.budget.clone();
        let rng = self.rng.clone();
        let command_queue = self.command_queue.clone();
        let outstanding_continuations = self.outstanding_continuations.clone();
        let pending_external_values = self.pending_external_values.clone();
        let pending_budget_terminals = self.pending_budget_terminals.clone();

        if let Err(error) = self.feed_event_inner(event) {
            self.tree = tree;
            self.budget = budget;
            self.rng = rng;
            self.command_queue = command_queue;
            self.outstanding_continuations = outstanding_continuations;
            self.pending_external_values = pending_external_values;
            self.pending_budget_terminals = pending_budget_terminals;
            return Err(error);
        }
        Ok(())
    }

    fn feed_event_inner(&mut self, event: EngineEvent) -> Result<(), SchedulerError> {
        let branch = event.branch();
        let node = self
            .tree
            .get(branch)
            .ok_or(SchedulerError::UnknownBranch(branch))?;
        let accepts_event = match &event {
            EngineEvent::ValueScored { .. } => node.state().is_live(),
            EngineEvent::TokenSampled { .. } | EngineEvent::BranchExhausted { .. } => {
                node.state() == BranchState::Active
            }
        };
        if !accepts_event {
            return Err(SchedulerError::BranchNotActive(branch));
        }
        if matches!(event, EngineEvent::TokenSampled { .. })
            && self.scorer.is_none()
            && self.pending_external_values.contains(&branch)
        {
            return Err(SchedulerError::ValueScorePending(branch));
        }

        let mut emitted = Vec::new();
        let mut tree_budget_exhausted = false;
        match &event {
            EngineEvent::TokenSampled {
                branch, logprob, ..
            } => {
                let projected = self.tree.projected_after_token(*branch, *logprob)?;
                let score = self.scorer.as_ref().map(|scorer| scorer.score(&projected));
                if score.is_some_and(|score| !score.is_finite()) {
                    return Err(SchedulerError::InvalidNumber("value score"));
                }
                let branch_tokens = self
                    .tree
                    .get(*branch)
                    .ok_or(SchedulerError::UnknownBranch(*branch))?
                    .tokens_generated();
                let outcome = self.budget.consume(*branch, branch_tokens)?;
                self.tree.record_token(*branch, *logprob)?;
                if let Some(score) = score {
                    self.tree.record_value(*branch, score)?;
                } else {
                    self.tree.mark_value_pending(*branch)?;
                    self.pending_external_values.insert(*branch);
                }
                self.outstanding_continuations.remove(branch);
                let external_score_pending = self.scorer.is_none();
                tree_budget_exhausted = outcome.tree_exhausted && !external_score_pending;
                if outcome.tree_exhausted && external_score_pending {
                    self.pending_budget_terminals
                        .insert(*branch, PendingBudgetTerminal::Tree);
                } else if outcome.branch_exhausted && external_score_pending {
                    self.pending_budget_terminals
                        .insert(*branch, PendingBudgetTerminal::Branch);
                } else if outcome.branch_exhausted && !outcome.tree_exhausted {
                    self.tree.finalize(*branch)?;
                    emitted.push(Command::Finalize { branch: *branch });
                    emitted.extend(self.reclaim_completed_ancestors(*branch)?);
                }
            }
            EngineEvent::BranchExhausted { branch } => {
                self.outstanding_continuations.remove(branch);
                self.pending_external_values.remove(branch);
                self.pending_budget_terminals.remove(branch);
                self.tree.finalize(*branch)?;
                emitted.push(Command::Finalize { branch: *branch });
                emitted.extend(self.reclaim_completed_ancestors(*branch)?);
            }
            EngineEvent::ValueScored { branch, score } => {
                if self.scorer.is_some() || !self.pending_external_values.contains(branch) {
                    return Err(SchedulerError::UnexpectedValueScore(*branch));
                }
                self.tree.record_value(*branch, *score)?;
                self.pending_external_values.remove(branch);
                match self.pending_budget_terminals.remove(branch) {
                    Some(PendingBudgetTerminal::Tree) => tree_budget_exhausted = true,
                    Some(PendingBudgetTerminal::Branch) => {
                        self.tree.finalize(*branch)?;
                        emitted.push(Command::Finalize { branch: *branch });
                        emitted.extend(self.reclaim_completed_ancestors(*branch)?);
                    }
                    None => {}
                }
            }
        }

        if tree_budget_exhausted {
            emitted.extend(self.terminate_tree(KillReason::TreeBudgetExhausted)?);
            self.pending_external_values.clear();
            self.pending_budget_terminals.clear();
            self.enqueue_commands(emitted);
            return Ok(());
        }

        emitted.extend(self.speculative_prune()?);
        let tree_before_policy = self.tree.clone();
        let policy_commands = self
            .policy
            .on_event(&event, &mut self.tree, &mut self.rng)?;
        self.validate_policy_commands(&tree_before_policy, &policy_commands)?;
        let terminal_branches: Vec<_> = policy_commands
            .iter()
            .filter_map(|command| match command {
                Command::Kill { branch, .. } | Command::Finalize { branch } => Some(*branch),
                Command::ForkAt { .. } | Command::Continue { .. } => None,
            })
            .collect();
        emitted.extend(policy_commands);
        for branch in terminal_branches {
            emitted.extend(self.reclaim_completed_ancestors(branch)?);
        }
        self.pending_external_values.retain(|branch| {
            self.tree
                .get(*branch)
                .is_some_and(|node| node.state().is_live())
        });
        self.pending_budget_terminals.retain(|branch, _| {
            self.tree
                .get(*branch)
                .is_some_and(|node| node.state().is_live())
        });
        self.enqueue_commands(emitted);
        Ok(())
    }

    fn validate_policy_commands(
        &self,
        tree_before_policy: &BranchTree,
        commands: &[Command],
    ) -> Result<(), SchedulerError> {
        let mut expected = tree_before_policy.clone();
        for command in commands {
            match *command {
                Command::ForkAt { branch, width } => {
                    let _ = expected.fork(branch, width)?;
                }
                Command::Kill { branch, reason } => expected.kill(branch, reason)?,
                Command::Finalize { branch } => expected.finalize(branch)?,
                Command::Continue { branch } => {
                    let node = expected
                        .get(branch)
                        .ok_or(SchedulerError::UnknownBranch(branch))?;
                    if node.state() != BranchState::Active {
                        return Err(SchedulerError::BranchNotActive(branch));
                    }
                }
            }
        }
        if !expected.has_same_structure(&self.tree) {
            return Err(SchedulerError::PolicyCommandTreeMismatch);
        }
        Ok(())
    }

    #[must_use]
    pub fn poll_commands(&mut self) -> Vec<Command> {
        self.command_queue.drain(..).collect()
    }

    pub fn drain(&mut self) -> Result<(), SchedulerError> {
        let commands = self.terminate_tree(KillReason::Drained)?;
        self.outstanding_continuations.clear();
        self.pending_external_values.clear();
        self.pending_budget_terminals.clear();
        self.enqueue_commands(commands);
        Ok(())
    }

    fn enqueue_commands(&mut self, commands: impl IntoIterator<Item = Command>) {
        for command in commands {
            match command {
                Command::ForkAt { branch, width } => {
                    self.remove_queued_continue(branch);
                    self.command_queue
                        .push_back(Command::ForkAt { branch, width });
                }
                Command::Kill { branch, reason } => {
                    self.remove_queued_continue(branch);
                    self.command_queue
                        .push_back(Command::Kill { branch, reason });
                }
                Command::Finalize { branch } => {
                    self.remove_queued_continue(branch);
                    self.command_queue.push_back(Command::Finalize { branch });
                }
                Command::Continue { branch } => {
                    let branch_can_advance = self.tree.get(branch).is_some_and(|node| {
                        node.state() == BranchState::Active
                            && node.tokens_generated() < self.budget.per_branch_limit()
                    });
                    let reserved =
                        u64::try_from(self.outstanding_continuations.len()).unwrap_or(u64::MAX);
                    if branch_can_advance
                        && !self.pending_external_values.contains(&branch)
                        && !self.outstanding_continuations.contains(&branch)
                        && reserved < self.budget.remaining_total()
                    {
                        self.outstanding_continuations.insert(branch);
                        self.command_queue.push_back(Command::Continue { branch });
                    }
                }
            }
        }
    }

    fn remove_queued_continue(&mut self, branch: BranchId) {
        self.outstanding_continuations.remove(&branch);
        self.command_queue.retain(
            |command| !matches!(command, Command::Continue { branch: queued } if *queued == branch),
        );
    }

    fn speculative_prune(&mut self) -> Result<Vec<Command>, SchedulerError> {
        let Some(margin) = self.speculative_kill_margin else {
            return Ok(Vec::new());
        };
        let scored: Vec<_> = self
            .tree
            .active_frontier()
            .into_iter()
            .filter(|branch| {
                self.tree
                    .get(*branch)
                    .is_some_and(crate::BranchNode::has_value)
            })
            .collect();
        let Some(best) = scored.iter().copied().min_by(|left, right| {
            let left_node = self.tree.get(*left).expect("known frontier branch");
            let right_node = self.tree.get(*right).expect("known frontier branch");
            right_node
                .value_estimate()
                .total_cmp(&left_node.value_estimate())
                .then_with(|| left.cmp(right))
        }) else {
            return Ok(Vec::new());
        };
        let best_value = self
            .tree
            .get(best)
            .expect("known frontier branch")
            .value_estimate();
        let mut victims: Vec<_> = scored
            .into_iter()
            .filter(|branch| {
                *branch != best
                    && best_value
                        - self
                            .tree
                            .get(*branch)
                            .expect("known frontier branch")
                            .value_estimate()
                        > margin
            })
            .collect();
        victims.sort_unstable();

        let mut commands = Vec::new();
        for victim in victims {
            self.tree.kill(victim, KillReason::SpeculativeKill)?;
            commands.push(Command::Kill {
                branch: victim,
                reason: KillReason::SpeculativeKill,
            });
            commands.extend(self.reclaim_completed_ancestors(victim)?);
        }
        Ok(commands)
    }

    fn reclaim_completed_ancestors(
        &mut self,
        branch: BranchId,
    ) -> Result<Vec<Command>, SchedulerError> {
        let mut commands = Vec::new();
        let mut current = self.tree.parent(branch)?;
        while let Some(parent) = current {
            let parent_state = self
                .tree
                .get(parent)
                .ok_or(SchedulerError::UnknownBranch(parent))?
                .state();
            if parent_state.is_terminal() {
                current = self.tree.parent(parent)?;
                continue;
            }
            let all_children_terminal = self.tree.children(parent)?.iter().all(|child| {
                self.tree
                    .get(*child)
                    .is_some_and(|node| node.state().is_terminal())
            });
            if !all_children_terminal {
                break;
            }
            self.tree.kill(parent, KillReason::AncestorReclaimed)?;
            commands.push(Command::Kill {
                branch: parent,
                reason: KillReason::AncestorReclaimed,
            });
            current = self.tree.parent(parent)?;
        }
        Ok(commands)
    }

    fn terminate_tree(&mut self, reason: KillReason) -> Result<Vec<Command>, SchedulerError> {
        let mut commands = Vec::new();
        let mut active = self.tree.active_frontier();
        active.sort_by(|left, right| {
            let left_node = self.tree.get(*left).expect("known frontier branch");
            let right_node = self.tree.get(*right).expect("known frontier branch");
            let left_score = if left_node.has_value() {
                left_node.value_estimate()
            } else {
                left_node.cumulative_logprob()
            };
            let right_score = if right_node.has_value() {
                right_node.value_estimate()
            } else {
                right_node.cumulative_logprob()
            };
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.cmp(right))
        });

        let best = active.first().copied();
        let mut victims: Vec<_> = active.into_iter().skip(1).collect();
        victims.sort_unstable();
        for victim in victims {
            self.tree.kill(victim, reason)?;
            commands.push(Command::Kill {
                branch: victim,
                reason,
            });
        }
        if let Some(best) = best {
            self.tree.finalize(best)?;
            commands.push(Command::Finalize { branch: best });
        }

        loop {
            let mut reclaimable: Vec<_> = self
                .tree
                .iter()
                .filter(|node| {
                    node.state() == BranchState::Expanded
                        && node.children().iter().all(|child| {
                            self.tree
                                .get(*child)
                                .is_some_and(|child_node| child_node.state().is_terminal())
                        })
                })
                .map(|node| (node.depth(), node.id()))
                .collect();
            if reclaimable.is_empty() {
                break;
            }
            reclaimable.sort_by(|left, right| right.cmp(left));
            for (_, branch) in reclaimable {
                if self
                    .tree
                    .get(branch)
                    .is_some_and(|node| node.state() == BranchState::Expanded)
                {
                    self.tree.kill(branch, reason)?;
                    commands.push(Command::Kill { branch, reason });
                }
            }
        }
        Ok(commands)
    }
}
